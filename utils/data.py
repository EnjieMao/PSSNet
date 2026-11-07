"""
utils/data.py — 数据集加载与预处理。

功能：
  - 按 trainval.txt / test.txt 划分读取红外图像与掩码（dataset/<数据集名>/）
  - 训练端数据增强（随机翻转/旋转/裁剪等）
  - 输出 (image, mask) 对，供 train.py 的 DataLoader 使用

支持数据集：NUAA-SIRST（256x256）、NUDT-SIRST（256x256）、IRSTD-1k（512x512）。
"""
import torch
import torch.nn.functional as F
import torch.utils.data as Data

import os
from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage import rotate
import os.path as osp
import sys
import random
import numpy as np
import cv2


def _compute_scale_target(mask, scale_max=4.0):
    """
    从单张二值 mask 计算 scale supervision target。
    每个前景像素记录所属连通域 sqrt(area) 的 log 归一化值。
    mask: numpy array (H, W), 值在 [0, 1]
    返回: numpy array (H, W), float32
    """
    img = (mask > 0.5).astype(np.uint8)
    num_labels, labeled = cv2.connectedComponents(img, connectivity=4)
    target = np.zeros_like(mask, dtype=np.float32)
    if num_labels <= 1:
        return target
    areas = np.bincount(labeled.ravel(), minlength=num_labels)
    scale_values = np.zeros(num_labels, dtype=np.float32)
    for j in range(1, num_labels):
        v = float(np.log1p(np.sqrt(float(areas[j]))) / scale_max)
        scale_values[j] = min(v, 1.0)
    return scale_values[labeled]


def _compute_q_soft(mask):
    """
    Q soft target: m * (1 - U(AvgPool2(m))).
    mask: numpy array (H, W)
    返回: numpy array (H, W), float32
    """
    m = torch.from_numpy((mask > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    occ = F.avg_pool2d(m, kernel_size=2)
    m_bar = F.interpolate(occ, size=m.shape[-2:], mode='nearest')
    q = (m * (1.0 - m_bar)).squeeze().numpy()
    return q.astype(np.float32)


def _compute_q_hard(mask):
    """
    Q hard target: m * (1 - U(MaxPool2(m))).
    mask: numpy array (H, W)
    返回: numpy array (H, W), float32
    """
    m = torch.from_numpy((mask > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m_next = F.max_pool2d(m, kernel_size=2)
    m_next_up = F.interpolate(m_next, size=m.shape[-2:], mode='nearest')
    q = (m * (1.0 - m_next_up)).squeeze().numpy()
    return q.astype(np.float32)


class TrainDataset(Data.Dataset):
    def __init__(self, args, mode='train', img_norm_cfg=None):
        self.args = args
        dataset_dir = args.dataset_root + '/' + args.dataset

        if args.dataset in ['Maritime_sirst', 'dataset']:
            if mode == 'train':
                self.imgs_dir = osp.join(dataset_dir, 'trainval', 'images')
                self.label_dir = osp.join(dataset_dir, 'trainval', 'masks')
            else:
                self.imgs_dir = osp.join(dataset_dir, 'detect', 'images')
                self.label_dir = osp.join(dataset_dir, 'detect', 'masks')

            self.names = []
            for filename in os.listdir(self.imgs_dir):
                if filename.endswith('png'):
                    base_name, _ = osp.splitext(filename)
                    self.names.append(base_name)
        else:
            if mode == 'train':
                txtfile = 'trainval.txt'
            elif mode == 'val' or mode == 'vis':
                txtfile = 'test.txt'

            self.list_dir = osp.join(dataset_dir, txtfile)
            self.imgs_dir = osp.join(dataset_dir, 'images')
            self.label_dir = osp.join(dataset_dir, 'masks')

            self.names = []
            with open(self.list_dir, 'r') as f:
                self.names += [line.strip() for line in f.readlines()]

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(args.dataset, dataset_dir)

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size

        # 全量预加载到内存：数据集很小（NUDT 663 张约 90MB），
        # 避免每个 worker 每轮重复 PNG 解码（实测解码是训练速度瓶颈）。
        # fork 出的 DataLoader worker 通过 copy-on-write 共享，不额外占内存。
        self._cache = None
        if self.mode in ('train', 'val'):
            self._cache = []
            for name in self.names:
                img_path = osp.join(self.imgs_dir, name + '.png')
                label_path = osp.join(self.label_dir, name + '_pixels0.png') \
                    if self.args.dataset == 'NUAA-SIRST' else osp.join(self.label_dir, name + '.png')
                if not osp.exists(label_path):
                    alt = osp.join(self.label_dir, name + '.png')
                    if osp.exists(alt):
                        label_path = alt
                    else:
                        alt = osp.join(self.label_dir, name + '_pixels0.png')
                        if osp.exists(alt):
                            label_path = alt
                img = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
                mask = np.array(Image.open(label_path), dtype=np.uint8)
                if img.shape[:2] != mask.shape[:2]:
                    # 兼容 image/mask 尺寸不一致的样本（如 Misc_111）
                    mask = np.array(Image.fromarray(mask).resize(
                        (img.shape[1], img.shape[0]), Image.NEAREST), dtype=np.uint8)
                self._cache.append((img, mask))

    def __getitem__(self, i):
        name = self.names[i]
        if self._cache is not None:
            img_np, mask_np = self._cache[i]
            img = Image.fromarray(img_np)
            mask = Image.fromarray(mask_np)
        else:
            img_path = osp.join(self.imgs_dir, name + '.png')
            if self.args.dataset == 'NUAA-SIRST':
                label_path = osp.join(self.label_dir, name + '_pixels0.png')
            else:
                   label_path = osp.join(self.label_dir, name + '.png')
            img = Image.open(img_path).convert('L')
            if not osp.exists(label_path):
                # 兼容两种 mask 命名：{name}_pixels0.png 与 {name}.png
                alt_label_path = osp.join(self.label_dir, name + '.png')
                if osp.exists(alt_label_path):
                    label_path = alt_label_path
                else:
                    alt_label_path = osp.join(self.label_dir, name + '_pixels0.png')
                    if osp.exists(alt_label_path):
                        label_path = alt_label_path
            mask = Image.open(label_path)
            if img.size != mask.size:
                # 兼容 image/mask 尺寸不一致的样本（如 Misc_111）
                mask = mask.resize(img.size, Image.NEAREST)
            img_np = np.array(img, dtype=np.uint8)
            mask_np = np.array(mask, dtype=np.uint8)
        if self.mode == 'train':
            # 快速通道：worker 只做 crop/flip/scale target，归一化与 q target 在 GPU 完成。
            # 返回 uint8 图像/mask + fp16 scale target，每 batch 传输量从 ~40MB 降到 ~2MB。
            img_u8, mask_u8 = self._random_crop(img_np, mask_np, self.crop_size, pos_prob=0.5)
            img_u8, mask_u8 = self._sync_transform(img_u8, mask_u8)

            img_batch = torch.from_numpy(np.ascontiguousarray(img_u8[np.newaxis, :]))
            mask_batch = torch.from_numpy(np.ascontiguousarray(mask_u8[np.newaxis, :]))
            s_target = _compute_scale_target(mask_u8.astype(np.float32) / 255.0, scale_max=4.0)
            s_target = torch.from_numpy(s_target[np.newaxis, :]).to(torch.float16)
            return img_batch, mask_batch, s_target, None, None

        elif self.mode == 'val':
            img = (np.array(img, dtype=np.float32) - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            mask = np.array(mask, dtype=np.float32) / 255.0
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]

            # 默认保持原始尺寸验证/测试；对 IRSTD-1k 512->256 等场景可显式开启 val-resize
            if getattr(self.args, 'val_resize', False):
                target_size = (self.base_size, self.base_size)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

            img = PadImg(img)
            mask = PadImg(mask)

            img, mask = img[np.newaxis, :], mask[np.newaxis, :]

            img = torch.from_numpy(np.ascontiguousarray(img)).to(torch.float32)
            mask = torch.from_numpy(np.ascontiguousarray(mask)).to(torch.float32)
            if img.shape != mask.shape:
                print('img!=mask in dataset')
            # val 不需要预计算 target
            return img, mask, None, None, None

        else:
            raise ValueError("Unkown self.mode")

    def __len__(self):
        return len(self.names)

    def _random_crop(self, img, mask, patch_size, pos_prob=None):
        h, w = img.shape
        if min(h, w) < patch_size:
            img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                         mode='constant')
            mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                          mode='constant')
            h, w = img.shape

        while 1:
            h_start = random.randint(0, h - patch_size)
            h_end = h_start + patch_size
            w_start = random.randint(0, w - patch_size)
            w_end = w_start + patch_size

            img_patch = img[h_start:h_end, w_start:w_end]
            mask_patch = mask[h_start:h_end, w_start:w_end]

            if pos_prob == None or random.random() > pos_prob:
                break
            elif mask_patch.sum() > 0:
                break

        return img_patch, mask_patch

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:  # 水平反转
            img = img[::-1, :]
            mask = mask[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            img = img[:, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            img = img.transpose(1, 0)
            mask = mask.transpose(1, 0)
        # random rotate
        if random.random() < 0.5:
            angle = random.uniform(-3, 3)
            img = rotate(img, angle, reshape=False, order=1)  # bilinear
            mask = rotate(mask, angle, reshape=False, order=0)  # nearest

        return img, mask


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1k':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'IRSTD-NUDT':
        # IRSTD-1k + NUDT-SIRST 融合数据集（trainval 1463 张实测统计）
        img_norm_cfg = dict(mean=90.900675, std=58.698318)
    elif dataset_name == 'NUAA-NUDT':
        # NUAA-SIRST + NUDT-SIRST 融合数据集（trainval 876 张实测统计）
        img_norm_cfg = dict(mean=109.347489, std=56.575327)
    elif dataset_name == 'SIRST2':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST3':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'NUDT-SIRST-Sea-Light':
        img_norm_cfg = dict(mean=21.39715576171875, std=10.919337272644043)
    elif dataset_name == 'Maritime_sirst':
        img_norm_cfg = dict(mean=36.6230583190918, std=13.484057426452637)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST5':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST6':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST7':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'IRDST-real':
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        with open(dataset_dir + '/' + '/trainval' + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + '/test' + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg
