"""
train.py — GL-Net 统一训练/测试入口。

功能：
  - 解析命令行参数与 --model-config（JSON 消融配置，见 configs/ 目录）
  - 构建 PSS-Net（model/pss_net.py），按配置开关 ADF/NSR/FDN/PCER 等模块
  - 训练循环：主分割损失（DiceFocalBCELoss）+ 深监督 + 双先验监督（GlobalLockLoss）
  - 优化器：Adam + GradualWarmup + CosineAnnealing；epoch 300 起评测，按 IoU 保存最优权重
  - 输出：weight/<数据集>/PSSNet-<实验名>-<时间戳>/ 下的 checkpoint、metric.log、源码快照

用法示例：
  python train.py --mode train --dataset NUAA-SIRST --dataset-root ./dataset \
      --batch-size 8 --epochs 800 --lr 0.001 --ngpu 0 \
      --model-config configs/paper_abl_5_full.json --exp-name MyExp --test-start-epoch 300
"""
from utils.data import *
from utils.metric_basic import *
from argparse import ArgumentParser
import torch
import torch.utils.data as Data
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
from model.pss_net import PSSNet
from model.loss import *
from torch.optim import Adagrad
from torch.optim import lr_scheduler
from utils.warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import os.path as osp
import os
import sys
import time
import cv2
import shutil
import glob
import json
import logging


def train_collate_fn(batch):
    """
    训练集自定义 collate：将 worker 返回的 (img, mask, s_target, None, None)
    中的有效张量堆叠，None 保留，避免 default_collate 无法处理 None。
    """
    imgs, masks, s_targets, _, _ = zip(*batch)
    return (torch.stack(imgs, dim=0),
            torch.stack(masks, dim=0),
            torch.stack(s_targets, dim=0),
            None, None)


def val_collate_fn(batch):
    """
    验证集 batch 的自定义 collate：把不同尺寸的图像 pad 到当前 batch 最大尺寸，
    并记录每张图的原始尺寸，供 PD_FA 计算真实面积。
    """
    imgs, masks, _, _, _ = zip(*batch)
    max_h = max(img.shape[-2] for img in imgs)
    max_w = max(img.shape[-1] for img in imgs)

    padded_imgs = []
    padded_masks = []
    orig_sizes = []
    for img, mask in zip(imgs, masks):
        h, w = img.shape[-2], img.shape[-1]
        pad_h = max_h - h
        pad_w = max_w - w
        padded_imgs.append(F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0))
        padded_masks.append(F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0))
        orig_sizes.append((h, w))

    return torch.stack(padded_imgs, dim=0), torch.stack(padded_masks, dim=0), orig_sizes


class Tee:
    """同时向多个流写入，并立即刷新，用于把 tqdm/print 实时落盘。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # 让 tqdm 在后台也能启用进度条
        return False


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')
    parser.add_argument('--dataset-root', type=str, default='./dataset', help='root directory')
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name:  NUDT-SIRST(50:50/256), IRSTD-1k(80:20/512), NUAA-SIRST(50:50/256)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1500,
                        help='It often takes 1000+ epochs with the Transformer to achieve complete convergence')
    parser.add_argument('--lr', type=float, default=0.001, help='adam0.001,adagrad0.05')
    parser.add_argument('--warm-epoch', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=0, help='GPU number')

    parser.add_argument('--base-size', type=int, default=None)  # IRSTD-1K 512
    parser.add_argument('--crop-size', type=int, default=None)  # IRSTD-1K 512
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train', help='train or test')  # The test can be conducted by directly modifying the parameters.
    parser.add_argument('--weight-path', type=str, default='./weight/NUAA-SIRST/weight-NUAA-SIRST.pkl', help='path to weights')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='path to checkpoint.pkl for resume training')
    parser.add_argument('--model-config', type=str, default=None, help='JSON string or JSON file path for model ablation config')
    parser.add_argument('--val-resize', action='store_true',
                        help='验证/测试时把图像缩放到 base_size（用于 IRSTD-1k 512->256 等场景）')
    parser.add_argument('--exp-name', type=str, default=None, help='experiment name suffix for save folder')
    parser.add_argument('--test-start-epoch', type=int, default=300, help='start validation/test from this epoch')
    parser.add_argument('--checkpoint-every', type=int, default=50,
                        help='save checkpoint every N epochs (besides best model), 0 to disable periodic checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--amp', action='store_true',
                        help='启用 AMP 混合精度训练（前向 autocast + GradScaler），提速约 1.5-2x')
    parser.add_argument('--val-every', type=int, default=1,
                        help='test-start-epoch 之后每 N 轮评测一次（默认 1，加速可设 2）')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile 编译模型（首轮编译需 1-2 分钟，之后明显提速）')
    args = parser.parse_args()
    # 根据dataset自动设置输入尺寸
    if args.dataset == 'IRSTD-1k':
        args.base_size = 512 if args.base_size is None else args.base_size
        args.crop_size = 512 if args.crop_size is None else args.crop_size
    elif args.dataset in ['NUAA-SIRST', 'NUDT-SIRST']:
        args.base_size = 256 if args.base_size is None else args.base_size
        args.crop_size = 256 if args.crop_size is None else args.crop_size
    else:
        args.base_size = 256 if args.base_size is None else args.base_size
        args.crop_size = 256 if args.crop_size is None else args.crop_size
    return args


def parse_model_config(config_arg):
    """解析 --model-config 参数：支持 JSON 字符串或 JSON 文件路径。"""
    if config_arg is None:
        return {}
    config_arg = config_arg.strip()
    if osp.exists(config_arg):
        with open(config_arg, 'r') as f:
            return json.load(f)
    try:
        return json.loads(config_arg)
    except json.JSONDecodeError as e:
        raise ValueError('--model-config 必须是合法 JSON 字符串或 JSON 文件路径: %s' % e)


def seed_worker(seed):
    """DataLoader worker 初始化：让每个 worker 使用独立但确定的种子。"""
    def fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return fn


def seed_pytorch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        # 根据传入参数设置随机种子
        seed_pytorch(args.seed)

        self.args = args
        self.start_epoch = 0
        self.mode = args.mode
        self.warm_epoch = args.warm_epoch

        trainset = TrainDataset(args, mode='train')
        valset = TrainDataset(args, mode='val')
        self.img_norm_cfg = trainset.img_norm_cfg

        # 先解析配置，便于根据配置调整 val_loader batch size
        model_config = parse_model_config(args.model_config)
        self.model_config = model_config
        self.arch = model_config.get('arch', 'pss')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True, num_workers=4,
                                            pin_memory=True, persistent_workers=False,
                                            collate_fn=train_collate_fn)
        val_batch_size = model_config.get('val_batch_size', 1)
        val_num_workers = 4 if val_batch_size > 1 else 0
        self.val_loader = Data.DataLoader(
            valset, val_batch_size, shuffle=False, drop_last=False,
            num_workers=val_num_workers, pin_memory=False,
            collate_fn=val_collate_fn,
            persistent_workers=val_num_workers > 0,
        )

        device = torch.device(f'cuda:{args.ngpu}' if torch.cuda.is_available() and
                                                     args.ngpu < torch.cuda.device_count() else 'cpu')
        self.device = device

        if self.arch != 'pss':
            raise RuntimeError('Only arch="pss" (PSS-Net) is supported.')
        model = PSSNet(args.crop_size, 1, config=model_config)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use ' + str(torch.cuda.device_count()) + ' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(self.device)
        if getattr(args, 'compile', False):
            # torch.compile：小模型 kernel-launch 开销占比高，编译后前向约 5x
            model = torch.compile(model)
        self.model = model

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs - self.warm_epoch,
                                                          eta_min=1e-5)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.warm_epoch,
                                                after_scheduler=scheduler_cosine)
        # AMP：--amp 开启混合精度训练
        self.use_amp = getattr(args, 'amp', False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)


        use_distance_bce = model_config.get('use_distance_bce', False)
        dist_sigma = model_config.get('dist_sigma', 2.0)
        self.loss_fun = DiceFocalBCELoss(
            dice_weight=model_config.get('dice_weight', 2.0),
            focal_weight=model_config.get('focal_weight', 1.0),
            bce_weight=model_config.get('bce_weight', 1.0),
            use_distance_bce=use_distance_bce, dist_sigma=dist_sigma,
        )
        gl_pos_weight = model_config.get('pos_weight', 1.0)
        gl_scale_weight = model_config.get('scale_weight', 1.0)
        gl_scale_gamma = model_config.get('scale_gamma', 3.0)
        gl_scale_mode = model_config.get('scale_mode', 'dist')
        self.gl_loss_fun = GlobalLockLoss(
            pos_weight=gl_pos_weight,
            scale_weight=gl_scale_weight,
            scale_gamma=gl_scale_gamma,
            scale_mode=gl_scale_mode,
        )
        # PSS-Net 分层先验监督（arch='pss' 时启用）
        if self.arch == 'pss':
            self.pss_loss_fun = PSSPriorLoss(
                pos_weight=gl_pos_weight,
                pos2_weight=model_config.get('pos2_weight', 0.5),
                scale_weight=gl_scale_weight,
                risk_weight=model_config.get('risk_weight', 1.0),
                survival_weight=model_config.get('survival_weight', 1.0),
                state_consistency_weight=model_config.get('state_consistency_weight', 0.0),
                scale_max=model_config.get('scale_max', 4.0),
                decoder_pos_weight=model_config.get('decoder_pos_weight', 1.0),
                decoder_pos_top_weight=model_config.get('decoder_pos_top_weight', 0.5),
                pcer_anchor_weight=model_config.get('pcer_anchor_weight', 0.0),
                pcer_anchor_start_epoch=model_config.get('pcer_anchor_start_epoch', 50),
                pcer_anchor_warmup_epochs=model_config.get('pcer_anchor_warmup_epochs', 20),
                risk_bg_weight=model_config.get('risk_bg_weight', 0.1),
                q_evidence_weight=model_config.get('q_evidence_weight', 0.3),
                state_activation_weight=model_config.get('state_activation_weight', 0.0),
                supervise_level2=model_config.get('supervise_level2', True),
            )
        else:
            self.pss_loss_fun = None
        self.load_balance_weight = model_config.get('load_balance_weight', 0.01)
        # v7-fix3: 禁用 fix2-hotfix 的 detail bias / router L2，使用纯 Q 驱动 Router
        self.detail_bias_weight = model_config.get('detail_bias_weight', 0.0)
        self.router_reg_weight = model_config.get('router_reg_weight', 0.0)
        self.PD_FA = PD_FA()
        self.mIoU = mIoU()
        self.ROC = ROCMetric(1, 10)
        self.F_metric = F_metric(nclass=1)
        self.best_fmeasure = 0
        self.best_iou = 0
        self.best_fa = 1e3

        if args.mode == 'train':
            if args.if_checkpoint:
                assert args.checkpoint_path is not None, '续训请通过 --checkpoint-path 指定 checkpoint.pkl 路径'
                print('Resume training from checkpoint: %s' % args.checkpoint_path)
                checkpoint = torch.load(args.checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_iou = checkpoint.get('iou', 0)
                self.best_fa = checkpoint.get('fa', 1e3)
                self.save_folder = osp.dirname(osp.abspath(args.checkpoint_path))
            else:
                suffix = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                if args.exp_name:
                    # 防止 exp_name 中的 '/' 创建额外子目录
                    safe_exp_name = args.exp_name.replace('/', '_').replace('\\', '_')
                    suffix = '%s-%s' % (safe_exp_name, suffix)
                self.save_folder = 'weight/%s/PSSNet-%s' % (args.dataset, suffix)
                if not osp.exists(self.save_folder):
                    os.makedirs(self.save_folder)
                self._save_source_code(osp.join(self.save_folder, 'src'))
                self._save_model_config(model_config, osp.join(self.save_folder, 'model_config.json'))
            # 训练日志实时落盘
            self.log_file = osp.join(self.save_folder, 'training.log')
            self.logger = logging.getLogger('Trainer')
            self.logger.setLevel(logging.INFO)
            # 避免重复 handler
            if not self.logger.handlers:
                fh = logging.FileHandler(self.log_file, mode='a')
                fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                self.logger.addHandler(fh)
                if sys.stdout.isatty():
                    sh = logging.StreamHandler(sys.stdout)
                    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                    self.logger.addHandler(sh)
        if args.mode == 'test':
            weight = torch.load(args.weight_path, map_location=f'cuda:{args.ngpu}', weights_only=False)
            self.model.load_state_dict(weight, strict=False)
            # checkpoint continue training: weight['state_dict'] / test:weight
            self.warm_epoch = -1

    def _save_model_config(self, config, path):
        """保存本次训练的模型配置到实验目录。"""
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def _save_source_code(self, src_dir):
        """训练启动时备份当前项目中的所有 .py 源文件到实验目录，便于复现实验。
        跳过 weight/、__pycache__/ 等目录，避免把历史实验权重和缓存源码重复备份。
        """
        if osp.exists(src_dir):
            shutil.rmtree(src_dir)
        os.makedirs(src_dir, exist_ok=True)

        project_root = osp.abspath(osp.dirname(__file__))
        skip_dirs = {'weight', '__pycache__', '.git', '.venv', 'venv', 'node_modules'}
        for py_file in glob.glob(osp.join(project_root, '**', '*.py'), recursive=True):
            rel_path = osp.relpath(py_file, project_root)
            # 如果相对路径中包含跳过目录，则忽略
            if any(part in skip_dirs for part in rel_path.split(osp.sep)):
                continue
            dst_path = osp.join(src_dir, rel_path)
            os.makedirs(osp.dirname(dst_path), exist_ok=True)
            shutil.copy2(py_file, dst_path)

    def train(self, epoch):
        self.model.train()
        lr = self.optimizer.param_groups[0]['lr']
        losses = AverageMeter()
        tag = True
        # 交互终端：正常 tqdm 进度条；后台运行：禁用 tqdm 进度条，
        # 改为每 5 步输出一行简洁进度，避免日志文件被 \r 刷屏
        is_tty = sys.stdout.isatty()
        if is_tty:
            tbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', ncols=120,
                        dynamic_ncols=True, leave=True)
        else:
            tbar = self.train_loader
        for i, (data, mask, s_target, q_soft, q_hard) in enumerate(tbar):

            # 快速数据通道：img/mask 以 uint8 传入，GPU 上归一化；
            # q_soft/q_hard target 由 mask 在 GPU 上计算（公式与 CPU 版一致）
            data = data.to(self.device).float()
            data = (data - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            labels = mask.to(self.device).float() / 255.0
            if q_soft is None:
                m_ = labels
                q_soft = m_ * (1.0 - F.interpolate(F.avg_pool2d(m_, 2),
                                                   size=m_.shape[-2:], mode='nearest'))
                q_hard = m_ * (1.0 - F.interpolate(F.max_pool2d(m_, 2),
                                                   size=m_.shape[-2:], mode='nearest'))
            if s_target is not None:
                s_target = s_target.to(self.device).float()

            if self.arch == 'pss':
                with torch.autocast(device_type='cuda', enabled=self.use_amp):
                    masks, pred, pss_state = self.model(data, tag)
            else:
                with torch.autocast(device_type='cuda', enabled=self.use_amp):
                    masks, pred = self.model(data, tag)
                pss_state = None
            labels_original = labels  # keep full-resolution GT for prior supervision

            # 状态加权：P 决定关注哪里，Q 决定哪里高风险，组合成全分辨率权重图
            weight_map = None
            if self.arch == 'pss' and pss_state is not None:
                if pss_state.get('position') is not None and pss_state.get('risk') is not None:
                    # 用最高层 P/Q（全分辨率），上采样后保证目标区权重 > 1
                    p_full = torch.sigmoid(pss_state['position'][0])
                    q_full = torch.sigmoid(pss_state['risk'][0])
                    weight_map = 1.0 + p_full + q_full
                    weight_map = weight_map.detach()

            # 1) 主分割 loss + 深监督：只在这里做平均
            seg_loss = self.loss_fun(pred, labels, weight=weight_map)
            for j in range(len(masks)):
                # 深监督：低分辨率预测上采样回全分辨率后与原始 GT 计算，
                # 避免下采样 GT 导致 tiny target 在标签中消失
                m_j = masks[j]
                if m_j.shape[-2:] != labels_original.shape[-2:]:
                    m_j = F.interpolate(m_j, size=labels_original.shape[-2:],
                                        mode='bilinear', align_corners=False)
                seg_loss = seg_loss + self.loss_fun(m_j, labels_original, weight=weight_map)
            seg_loss = seg_loss / (len(masks) + 1)

            loss = seg_loss

            # 2) 先验监督：PSS-Net 分层 (P,S,Q) 或原 GlobalLock 双先验
            # 注意：PSS auxiliary losses 不再被深监督平均系数缩小
            if self.pss_loss_fun is not None:
                if pss_state.get('position') is not None:
                    # 把 DataLoader worker 预计算的 scale/q target 传到 loss
                    pss_loss, _ = self.pss_loss_fun(
                        pss_state['position'], pss_state['scale'],
                        pss_state['risk'], labels_original,
                        s_targets=s_target.to(self.device) if s_target is not None else None,
                        q_soft_targets=q_soft.to(self.device) if q_soft is not None else None,
                        q_hard_targets=q_hard.to(self.device) if q_hard is not None else None,
                        survival_pairs=pss_state.get('survival'),
                        decoder_position_maps=pss_state.get('decoder_position'),
                        reconcile=pss_state.get('reconcile'),
                        neck_influence=pss_state.get('neck_influence'),
                        epoch=epoch)
                    loss = loss + pss_loss
            else:
                # GlobalLock prior supervision
                position_map = getattr(self.model, 'position_map', None)
                scale_map = getattr(self.model, 'scale_map', None)
                if position_map is not None and scale_map is not None:
                    # 传递全分辨率 GT，由 GlobalLockLoss 内部做高斯热图 + bilinear 下采样
                    gl_loss, _, _ = self.gl_loss_fun(position_map, scale_map, labels_original)
                    loss = loss + gl_loss

            # 3) v7-fix2-hotfix: 阻止 Q-Router 把 gate 系统性压低
            if self.arch == 'pss':
                gates = pss_state.get('gates')
                if gates is not None:
                    detail_bias = 0.0
                    for g in gates:
                        detail_bias = detail_bias + F.relu(0.5 - g).mean()
                    loss = loss + self.detail_bias_weight * detail_bias

                # 对 router 的 feat_term 加轻量 L2，起步时 feat_term=0，
                # 训练中也保持 feat_term 只能微调而非主导 gate。
                router_reg = 0.0
                for name, param in self.model.named_parameters():
                    if 'router' in name and param.requires_grad:
                        router_reg = router_reg + (param ** 2).sum()
                loss = loss + self.router_reg_weight * router_reg

            # 4) MoE load balance loss
            lb_loss = getattr(self.model, 'load_balance_loss', None)
            if lb_loss is not None:
                loss = loss + self.load_balance_weight * lb_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.update(loss.item(), pred.size(0))
            if is_tty:
                tbar.set_postfix(loss=f'{losses.avg:.4f}', lr=f'{lr:.6f}')

        if is_tty:
            print('Epoch %d, lr:%f, loss %.4f' % (epoch, lr, losses.avg), flush=True)
        self.logger.info('Epoch %d, lr:%f, loss %.4f' % (epoch, lr, losses.avg))
        self.scheduler.step(epoch)

    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.F_metric.reset()
        # self.ROC.reset()
        is_tty = sys.stdout.isatty()
        # 交互终端：正常 tqdm；后台：简单枚举
        if is_tty:
            tbar = tqdm(self.val_loader)
        else:
            tbar = self.val_loader
        tag = False
        with torch.no_grad():
            for i, (data, mask, orig_sizes) in enumerate(tbar):
                data = data.to(self.device)
                mask = mask.to(self.device)

                with torch.autocast(device_type='cuda',
                                    enabled=getattr(self, 'use_amp', False)):
                    if self.arch == 'pss':
                        _, pred, _ = self.model(data, tag)
                    else:
                        _, pred = self.model(data, tag)

                self.mIoU.update((pred.sigmoid() > 0.5).cpu(), mask.cpu())
                self.F_metric.update((pred.sigmoid() > 0.5).cpu(), mask.cpu())

                # PD_FA 按单图处理，使用每张图的原始未 pad 尺寸，支持 val_batch_size > 1
                for b in range(pred.size(0)):
                    oh, ow = orig_sizes[b]
                    pred_b = (pred[b, 0, :, :].sigmoid() > 0.5).cpu()
                    mask_b = mask[b, 0, :, :].cpu()
                    # 裁剪掉 pad 区域
                    if pred_b.shape[-2] != oh or pred_b.shape[-1] != ow:
                        pred_b = pred_b[:oh, :ow]
                        mask_b = mask_b[:oh, :ow]
                    self.PD_FA.update(pred_b, mask_b, (oh, ow))

                results = self.mIoU.get()

                if is_tty:
                    tbar.set_description('Epoch %d, IoU %.4f' % (epoch, results[1]))
            results1 = self.mIoU.get()
            results2 = self.PD_FA.get()
            prec, recall, fmeasure = self.F_metric.get()
            # 每次评测一行汇总（非交互终端下 tqdm 已关闭，靠这行记录进度）
            log_msg = 'Epoch %d, IoU %.5f, F1 %.5f, PD %.5f, FA %.5f' % (
                epoch, results1[1], fmeasure, results2[0], results2[1] * 1000000)
            if hasattr(self, 'logger'):
                self.logger.info(log_msg)
            else:
                print(log_msg, flush=True)

            if self.mode == 'train':
                is_best = results1[1] > self.best_iou
                if is_best:
                    self.best_iou = results1[1]

                    torch.save(self.model.state_dict(),
                               self.save_folder + '/weight-%s.pkl' % (args.dataset))  # 训练时最优模型保存
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:05d}\t - IoU {:.5f}\t - F1 {:.5f}\t - PD {:.5f}\t - FA {:.5f}\n'.
                                format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                       epoch, self.best_iou, fmeasure, results2[0], results2[1] * 1000000))

                # 仅在性能提升或按指定间隔保存 checkpoint，避免每个 epoch 都写大文件导致 IO/空间压力
                save_ckpt = is_best or (args.checkpoint_every > 0 and epoch > 0 and
                                        epoch % args.checkpoint_every == 0)
                if save_ckpt:
                    all_states = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                                  "epoch": epoch, "fa": self.best_fa, "iou": self.best_iou}
                    torch.save(all_states, self.save_folder + '/checkpoint.pkl')
            elif self.mode == 'test':
                print("pixAcc, mIoU:\t" + str(results1), flush=True)
                print("PD, FA:\t" + str(results2), flush=True)
                print('F_measure:\t' + str(fmeasure), flush=True)


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)

    if trainer.mode == 'train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            if epoch >= args.test_start_epoch and \
                    (epoch - args.test_start_epoch) % args.val_every == 0:
                trainer.test(epoch)
    else:
        trainer.test(1)
