"""
model/loss.py — GL-Net 损失函数集合。

当前训练流程使用的类：
  - DiceFocalBCELoss   : 主分割损失 = 2*Dice + Focal + BCE（或 DistBCE）
  - DistanceWeightedBCE: 距离加权 BCE，bce * (1 + exp(-d/sigma))，
                         d 为像素到最近 GT 前景的欧氏距离
  - GlobalLockLoss     : 双先验监督——position_map 对高斯热图 GT 的 MSE，
                         scale_map 对 GT √area 图的 MSE

历史遗留（当前训练未使用，保留以备对比实验）：
  - AdaFocalLoss, SoftIoULoss
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class AdaFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        """Adaptive parameter adjustment"""

    def forward(self, pred, target):
        # pred = torch.sigmoid(pred)
        # 计算面积自适应权重
        area_weight = self._get_area_weight(target)  # [N,1,1,1]
        smooth = 1

        intersection = pred.sigmoid() * target
        iou = (intersection.sum() + smooth) / (pred.sigmoid().sum() + target.sum() - intersection.sum() + smooth)
        iou = torch.clamp(iou, min=1e-6, max=1 - 1e-6).detach()  # .detach()根据收敛情况可以选择去除
        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # 自带sigmoid

        target = target.type(torch.long)
        at = target * area_weight + (1 - target) * (1 - area_weight)
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        F_loss = (1 - pt) ** (1 - iou + 1e-6) * BCE_loss

        F_loss = at * F_loss
        return F_loss.sum()

    def _get_area_weight(self, target):
        # 小目标增强权重
        area = target.sum(dim=(1, 2, 3))  # [N,1]
        return torch.sigmoid(1 - area / (area.max() + 1)).view(-1, 1, 1, 1)


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def IOU(self, pred, mask):
        smooth = 1

        intersection = pred * mask
        loss = (intersection.sum() + smooth) / (pred.sum() + mask.sum() - intersection.sum() + smooth)

        loss = 1 - torch.mean(loss)
        return loss

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        loss_iou = self.IOU(pred, mask)

        return loss_iou
class DiceFocalBCELoss(nn.Module):
    """
    Combined loss: 2 * DiceLoss + 1 * FocalLoss + 1 * BCELoss
    pred: raw logits (N, 1, H, W)
    target: binary mask (N, 1, H, W), float in {0, 1}

    Args:
        use_distance_bce: 是否用 DistanceWeightedBCE 替换标准 BCE。
        dist_sigma: DistanceWeightedBCE 的距离衰减系数。
    """
    def __init__(self, dice_weight=2.0, focal_weight=1.0, bce_weight=1.0, gamma=2.0, smooth=1.0,
                 use_distance_bce=False, dist_sigma=2.0):
        super(DiceFocalBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.gamma = gamma
        self.smooth = smooth
        self.use_distance_bce = use_distance_bce
        self.dist_sigma = dist_sigma
        if use_distance_bce:
            self._bce_fn = DistanceWeightedBCE(sigma=dist_sigma, base_weight=1.0)
        else:
            self._bce_fn = None

    def dice_loss(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum()
        dice = (2. * intersection + self.smooth) / (pred_prob.sum() + target.sum() + self.smooth)
        return 1 - dice
    def focal_loss(self, pred, target, weight=None):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        focal = (1 - pt) ** self.gamma * bce
        if weight is not None:
            focal = focal * weight
        return focal.mean()
    def bce_loss(self, pred, target, weight=None):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        if self.use_distance_bce:
            # DistanceWeightedBCE 已有内部 weight，此处不再重复乘
            return self._bce_fn(pred, target)
        if weight is not None:
            bce = bce * weight
        return bce.mean()
    def forward(self, pred, target, weight=None):
        loss = self.dice_weight * self.dice_loss(pred, target) \
               + self.focal_weight * self.focal_loss(pred, target, weight=weight) \
               + self.bce_weight * self.bce_loss(pred, target, weight=weight)
        return loss


class DistanceWeightedBCE(nn.Module):
    """
    Distance-weighted BCE loss.
    For false positives, reduce the penalty when they are close to the GT.
    pred: raw logits (N, 1, H, W)
    target: binary mask (N, 1, H, W), float in {0, 1}

    Note:
        The EDT is cached per batch. In the current training loop the same
        full-resolution GT is reused for the main prediction and all deep-
        supervision outputs, so the cached distance map is valid across those
        calls and avoids recomputing the EDT three/four times per iteration.
    """

    def __init__(self, sigma=2.0, base_weight=1.0):
        super().__init__()
        self.sigma = sigma
        self.base_weight = base_weight
        self._cache_key = None
        self._cache_dt = None

    def _distance_transform(self, mask):
        """Euclidean distance transform for a binary mask."""
        import scipy.ndimage as ndi
        B, _, H, W = mask.shape
        dt = torch.zeros_like(mask)
        mask_np = mask.cpu().numpy()
        for i in range(B):
            dt[i, 0] = torch.from_numpy(ndi.distance_transform_edt(mask_np[i, 0])).to(mask.device)
        return dt

    def forward(self, pred, target):
        # 基础 BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # 距离变换：每个像素到最近 GT 前景的距离。
        # 同一张 target 在当前 batch 内会被多次使用（pred + deep supervision），
        # 用 shape + data_ptr 作为缓存键复用同一份 distance map。
        with torch.no_grad():
            key = (tuple(target.shape), str(target.device), target.data_ptr())
            if self._cache_key != key or self._cache_dt is None:
                # 1 - target 表示背景，背景的距离变换 = 到最近前景的距离
                self._cache_dt = self._distance_transform(1 - target)
                self._cache_key = key
            dist_to_gt = self._cache_dt

        # 距离越近，权重越大；距离越远，权重越小
        weight = torch.exp(-dist_to_gt / self.sigma)

        # false positive (pred=1, target=0) 且离 GT 近的区域，减轻惩罚
        # 其他区域保持 base_weight
        weighted_bce = bce * (self.base_weight + weight)

        return weighted_bce.mean()


class GlobalLockLoss(nn.Module):
    """
    Supervision loss for the GlobalLock position and scale priors.

    Args:
        pos_weight: weight for position prior BCE loss.
        scale_weight: weight for scale prior MSE loss.
        scale_gamma: maximum scale value, should match GlobalAttentionLock.scale_gamma.
    """

    def __init__(self, pos_weight=1.0, scale_weight=1.0, scale_gamma=3.0, scale_mode='dist'):
        super().__init__()
        self.pos_weight = pos_weight
        self.scale_weight = scale_weight
        self.scale_gamma = scale_gamma
        self.scale_mode = scale_mode

    @staticmethod
    def _distance_transform(mask):
        """
        Compute Euclidean distance transform for a binary mask.
        mask: (N, 1, H, W) torch tensor
        returns: (N, 1, H, W) torch tensor
        """
        import scipy.ndimage as ndi
        B, _, H, W = mask.shape
        dt = torch.zeros_like(mask)
        mask_np = mask.cpu().numpy()
        for i in range(B):
            dt[i, 0] = torch.from_numpy(ndi.distance_transform_edt(mask_np[i, 0])).to(mask.device)
        return dt

    def _scale_target(self, mask):
        """
        Generate scale-map supervision from a binary mask.
        For each target region, the scale is proportional to the distance to the
        region boundary (largest at the target center, zero at background).
        """
        # distance inside target regions
        fg_dt = self._distance_transform(mask)
        # scale in [0, scale_gamma]
        max_per_sample = fg_dt.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        scale_target = fg_dt / max_per_sample * self.scale_gamma
        return scale_target

    def _scale_target_area(self, mask):
        """
        生成空间尺度图监督：每个像素记录其所属连通域面积的平方根。
        背景像素为 0，不同目标按自身大小编码。输出 shape (N, 1, H, W)。
        """
        import scipy.ndimage as ndi
        import numpy as np
        B, _, H, W = mask.shape
        target = torch.zeros(B, 1, H, W, device=mask.device, dtype=torch.float32)
        mask_np = mask.cpu().numpy()
        for i in range(B):
            labeled, num = ndi.label(mask_np[i, 0])
            if num == 0:
                continue
            for j in range(1, num + 1):
                area = (labeled == j).sum()
                scale_val = float(np.sqrt(area))
                target[i, 0][labeled == j] = scale_val
        return target

    @staticmethod
    def _gaussian_blur(mask, sigma=2.0):
        """
        Apply Gaussian blur to a binary mask to create a soft position target.
        mask: (N, 1, H, W) torch tensor
        returns: (N, 1, H, W) torch tensor, blurred and normalized to [0, 1]
        """
        import scipy.ndimage as ndi
        B, _, H, W = mask.shape
        blurred = torch.zeros_like(mask)
        mask_np = mask.cpu().numpy()
        for i in range(B):
            blur = ndi.gaussian_filter(mask_np[i, 0], sigma=sigma)
            # 归一化到 [0, 1]，防止模糊后峰值变小
            blur = blur / (blur.max() + 1e-6)
            blurred[i, 0] = torch.from_numpy(blur).to(mask.device)
        return blurred

    def forward(self, position_map, scale_map, target):
        """
        Args:
            position_map: (N, 1, Hb, Wb), sigmoid output at bottleneck resolution
            scale_map: (N, 1, Hb, Wb), predicted scale values at bottleneck resolution
            target: (N, 1, H, W), binary ground-truth mask at full resolution
        """
        target_size = position_map.shape[-2:]

        # Position prior: 先在全分辨率做高斯热图，再 bilinear 下采样。
        # 这样 3×3 小目标在下采样到 32×32 时仍能留下微弱的软监督信号，
        # 避免 nearest 二值 mask 直接消失。
        with torch.no_grad():
            pos_target_full = self._gaussian_blur(target, sigma=2.0)
            pos_target = F.interpolate(
                pos_target_full, size=target_size, mode='bilinear', align_corners=False
            )
        pos_loss = F.mse_loss(position_map, pos_target, reduction='mean')

        # Scale prior
        if self.scale_mode == 'area':
            # 空间尺度图监督：先用 adaptive_max_pool2d 把全分辨率二值 mask 下采样到 bottleneck，
            # 保留小目标存在性；再在 bottleneck 上计算连通域面积 √，与 scale_map 同数量级。
            with torch.no_grad():
                target_bottleneck = F.adaptive_max_pool2d(target, target_size)
                scale_target = self._scale_target_area(target_bottleneck)
            scale_loss = F.mse_loss(scale_map, scale_target, reduction='mean')
        else:
            # 原 GlobalLock：distance-transform 空间尺度图
            with torch.no_grad():
                scale_target = self._scale_target(target)
            scale_loss = F.mse_loss(scale_map, scale_target, reduction='mean')

        loss = self.pos_weight * pos_loss + self.scale_weight * scale_loss
        return loss, pos_loss, scale_loss


class PSSPriorLoss(nn.Module):
    """
    PSS-Net 目标状态监督（Position–Scale–Survival 状态动力学）。

    对每级 stage 的状态 Z_l = (P_l, S_l, Q_l) 计算：
      - L_P: BCE + Dice，P* 为 adaptive_max_pool 到该级分辨率的二值 mask
             （max pooling 保留小目标存在性）
      - L_S: 仅在目标区域计算的 SmoothL1，S 为**当前层级下的**等效尺度：
             S*_l = min(log(1 + sqrt(area_l)) / scale_max, 1)，
             area_l 为该级 mask M_l 上的连通域面积（随分辨率逐层演化），
             与 PSSHead 的 sigmoid 输出同口径（∈[0,1]）
      - L_Q: 目标区 SmoothL1，Q* = M_l ⊙ (1 - U(AvgPool2(M_l)))
             （Target Downsampling Risk：2x2 下采样后目标占空比被稀释的程度，
              无需人工新增标签，由分割 mask 自动生成）；
             另加小权重背景项把 Q 压向 0（背景占空比低本就该低风险），
             避免背景区不受约束漂移导致风险图失去目标特异性
      - L_surv: 反事实生存评估监督（网络开启生存感知转换时）。
             u_D/u_C 评估对应候选中**实际可解码的目标证据强度**，
             两者共用同一监督目标 M_l↓（MaxPool2(M_l)），差异只来自
             候选内容本身，与"事前"风险预测 Q 构成互补信息源；
             评估器不输入 Q，保证 gate_s/gate_none 对照无 Q 泄漏
      - L_stc: State Transition Consistency（state_consistency_weight > 0 时）。
             位置一致性由 **GT 构造的风险标签 Q*** 调节（Q* 低则 σ(P_l)↓
             应与 σ(P_{l+1}) 一致，Q* 高允许变化）——不使用预测 Q 加权，
             防止网络抬高 Q 逃避一致性监督；
             尺度一致性：S_{l+1} 应等于 S_l 经 2 倍下采样的等效尺度演化
             T(s) = log1p(expm1(s·scale_max)/2) / scale_max（先还原为实际
             等效尺度、除以 2、再映射回归一化空间，log 域精确减半）
      - L_DP: Decoder Position Reconstruction Loss（PCER decoder 开启时）。
             对 decoder 重建位置链 P_0^D..P_3^D 逐级做
             BCE + Dice（目标为对应分辨率的 M_l，max-pool 保小目标）。
             P_l^D 不 detach，梯度沿 decoder 正常传播。
             注意：**刻意不添加** ‖P_l^E − P_l^D‖ 之类的直接一致性项——
             那会把 encoder_rescue / decoder_correction 压向 0，
             破坏 PCER 的"分歧—纠错"机制。
      - L_anchor: PCER 三张交互响应图的弱语义锚点（pcer_anchor_weight > 0 时）。
             由位置状态构造软参考区域（全程 detach，只做辅助监督）：
                 t_cons = σ(P^E)·σ(C)，t_resc = ReLU(σ(P^E)−σ(C))，
                 t_corr = ReLU(σ(C)−σ(P^E))
             与对应响应图做 MSE。目的是防止三张图学成雷同的 attention map
             或发生角色交换；锚点只约束语义角色，不直接控制特征。

    Args:
        pos_weight / scale_weight / risk_weight: L_P / L_S / L_Q 权重。
        survival_weight: 反事实生存评估损失权重。
        state_consistency_weight: 状态跨层一致性损失权重（0 为关闭）。
        scale_max: 尺度归一化常数（log 尺度上限），全网络唯一定义处。
        decoder_pos_weight: L_DP 权重。
        pcer_anchor_weight: L_anchor 权重（建议 0.1，弱约束）。
        pcer_anchor_start_epoch: 锚点延迟启用轮次（默认 50）——训练早期
            E/C 状态未成形，锚点 t≈0 会把响应图拉向坍塌，须等状态头
            初步收敛后再引入角色约束。
        pcer_anchor_warmup_epochs: 锚点权重线性 warm-up 轮数（默认 20）。
        risk_bg_weight: Q 背景约束权重（默认 0.1）。
        q_evidence_weight: Q 的 hard evidence 项权重（默认 0.3）。
    """

    def __init__(self, pos_weight=1.0, pos2_weight=0.5, scale_weight=1.0, risk_weight=1.0,
                 survival_weight=1.0, state_consistency_weight=0.0,
                 scale_max=4.0, decoder_pos_weight=1.0, decoder_pos_top_weight=0.5,
                 pcer_anchor_weight=0.0,
                 pcer_anchor_start_epoch=50, pcer_anchor_warmup_epochs=20,
                 risk_bg_weight=0.1, q_evidence_weight=0.3,
                 state_activation_weight=0.0,
                 supervise_level2=True):
        super().__init__()
        self.pos_weight = pos_weight
        self.pos2_weight = pos2_weight
        self.scale_weight = scale_weight
        self.risk_weight = risk_weight
        self.survival_weight = survival_weight
        self.state_consistency_weight = state_consistency_weight
        self.scale_max = scale_max
        self.decoder_pos_weight = decoder_pos_weight
        self.decoder_pos_top_weight = decoder_pos_top_weight
        self.pcer_anchor_weight = pcer_anchor_weight
        self.pcer_anchor_start_epoch = pcer_anchor_start_epoch
        self.pcer_anchor_warmup_epochs = pcer_anchor_warmup_epochs
        self.risk_bg_weight = risk_bg_weight
        self.q_evidence_weight = q_evidence_weight
        self.state_activation_weight = state_activation_weight
        self.supervise_level2 = supervise_level2

    @staticmethod
    def _scale_target(mask, scale_max):
        """
        单级尺度监督图：每个像素记录所属连通域（在该级 mask 上）的
        min(log(1+sqrt(area))/scale_max, 1)，背景为 0。
        mask: (N,1,H,W) -> (N,1,H,W)。

        使用 cv2.connectedComponents 替换 scipy.ndimage.label，
        connectivity=4 与 ndi.label 默认连通性语义一致，速度更快。
        """
        import cv2
        import numpy as np
        B, _, H, W = mask.shape
        target = torch.zeros(B, 1, H, W, device=mask.device, dtype=torch.float32)
        mask_np = mask.detach().cpu().numpy()
        for i in range(B):
            # cv2.connectedComponents 返回 (num_labels, labels)，
            # num_labels 包含背景（label=0），因此前景编号从 1 开始。
            num_labels, labeled = cv2.connectedComponents(
                (mask_np[i, 0] > 0.5).astype(np.uint8), connectivity=4)
            if num_labels <= 1:
                continue
            areas = np.bincount(labeled.ravel(), minlength=num_labels)
            scale_values = np.zeros(num_labels, dtype=np.float32)
            for j in range(1, num_labels):
                v = float(np.log1p(np.sqrt(float(areas[j]))) / scale_max)
                scale_values[j] = min(v, 1.0)
            target[i, 0] = torch.from_numpy(scale_values[labeled]).to(mask.device)
        return target

    @staticmethod
    def _dice_loss(prob, target, smooth=1.0):
        intersection = (prob * target).sum()
        dice = (2. * intersection + smooth) / (prob.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, position_maps, scale_maps, risk_maps, target,
                s_targets=None, q_soft_targets=None, q_hard_targets=None,
                survival_pairs=None, decoder_position_maps=None, reconcile=None,
                neck_influence=None, epoch=None):
        """
        Args:
            position_maps: list of (N,1,Hl,Wl) position logits（每级 stage）
            scale_maps:    list of (N,1,Hl,Wl) scale 预测（sigmoid 后）
            risk_maps:     list of (N,1,Hl,Wl) survival risk 预测（sigmoid 后）
            target:        (N,1,H,W) 全分辨率二值 GT mask
            s_targets:     (N,1,H,W) 全分辨率 scale target（DataLoader 预计算），
                           为 None 时回退到内部 _scale_target（慢，用于兼容）
            q_soft_targets:(N,1,H,W) 全分辨率 Q soft target（预计算）
            q_hard_targets:(N,1,H,W) 全分辨率 Q hard target（预计算）
            survival_pairs: list of (u_D, u_C)，各级生存感知转换的反事实
                            证据保留强度评估（logits，半分辨率），可为 None
            decoder_position_maps: list of (N,1,Hl,Wl)，PCER decoder 重建
                            位置链 [P0D..P3D]（logits，含 bottleneck 级），
                            可为 None
            reconcile:     PCER 的 dict，含 'consensus' / 'encoder_rescue' /
                           'decoder_correction' / 'coarse_position' 三个
                           （或四个）逐级列表，可为 None
            neck_influence: list of (N,1,H,W) FiLM 调制幅度，可为 None
        Returns:
            (total, parts)：total 为加权和，parts 为各分项 dict
        """
        pos_loss = target.new_zeros(())
        scale_loss = target.new_zeros(())
        risk_loss = target.new_zeros(())
        n_levels = len(position_maps)
        mask_levels = []
        risk_gt_levels = []

        for lvl, (p, s, q) in enumerate(zip(position_maps, scale_maps, risk_maps)):
            size = p.shape[-2:]
            with torch.no_grad():
                m_l = F.adaptive_max_pool2d(target, size)
                m_l = (m_l > 0.5).float()

                # Scale target：优先使用 DataLoader 预计算的全分辨率 target，
                # 插值到当前 level 分辨率，避免训练主循环 CPU 阻塞。
                if s_targets is not None:
                    s_t_full = s_targets.to(m_l.device)
                    s_t = F.adaptive_max_pool2d(s_t_full, size)
                else:
                    s_t = self._scale_target(m_l, self.scale_max)

                # Q target：仅使用 q_soft（q_hard 在 MaxPool 后恒为 0，无意义）
                if q_soft_targets is not None:
                    q_soft_full = q_soft_targets.to(m_l.device)
                    q_soft = F.adaptive_max_pool2d(q_soft_full, size)
                else:
                    occ = F.avg_pool2d(m_l, kernel_size=2)
                    m_bar = F.interpolate(occ, size=size, mode='nearest')
                    q_soft = m_l * (1.0 - m_bar)
                q_t = q_soft
            mask_levels.append(m_l)
            risk_gt_levels.append(q_t)

            # H/8（level 2）上 pixel-wise GT 极不可靠，可以选择不给显式监督，
            # 让该层状态完全由下游任务自由演化。
            if lvl == 2 and not self.supervise_level2:
                continue

            prob = torch.sigmoid(p)
            lvl_pos_loss = F.binary_cross_entropy_with_logits(p, m_l) \
                + self._dice_loss(prob, m_l)
            # P2（最低分辨率）目标信号极弱，降低权重避免被背景淹没
            if lvl == 2:
                lvl_pos_loss = lvl_pos_loss * self.pos2_weight
            pos_loss = pos_loss + lvl_pos_loss

            denom = m_l.sum() + 1e-6
            scale_loss = scale_loss + (
                m_l * F.smooth_l1_loss(s, s_t, reduction='none')).sum() / denom
            # v7-fix3: Q head 输出 raw logits，直接用 BCEWithLogits 监督。
            # 去掉 SmoothL1 避免与 BCE 梯度冲突，让 Q 的概率校准更干净。
            risk_loss = risk_loss + (
                m_l * F.binary_cross_entropy_with_logits(q, q_t, reduction='none')
            ).sum() / denom
            # 背景区小权重压 0：让 Q 成为处处良定义的风险图（目标特异、背景低）
            # q 现在为 logits，背景项用 sigmoid(q)
            risk_loss = risk_loss + self.risk_bg_weight * ((1.0 - m_l) * torch.sigmoid(q)).mean()

        # 平均只除以实际受监督的层数
        n_supervised = n_levels - (1 if (n_levels > 2 and not self.supervise_level2) else 0)
        pos_loss = pos_loss / max(n_supervised, 1)
        scale_loss = scale_loss / max(n_supervised, 1)
        risk_loss = risk_loss / max(n_supervised, 1)

        # 反事实生存评估监督：u_D/u_C（logits）度量各候选中实际可解码的
        # 目标证据强度，两者共用同一目标 M_l↓，差异只来自候选内容本身，
        # 不与 Q* 的定义重叠
        surv_loss = target.new_zeros(())
        n_surv = 0
        if survival_pairs:
            for i, (u_d, u_c) in enumerate(survival_pairs):
                if i >= len(mask_levels):
                    break
                with torch.no_grad():
                    m_next = F.max_pool2d(mask_levels[i], kernel_size=2)
                surv_loss = surv_loss \
                    + F.binary_cross_entropy_with_logits(u_d, m_next) \
                    + self._dice_loss(torch.sigmoid(u_d), m_next) \
                    + F.binary_cross_entropy_with_logits(u_c, m_next) \
                    + self._dice_loss(torch.sigmoid(u_c), m_next)
                n_surv += 1
        if n_surv > 0:
            surv_loss = surv_loss / n_surv

        # State Transition Consistency：位置（Q 调节）+ 尺度（log 域减半演化）
        stc_loss = target.new_zeros(())
        if self.state_consistency_weight > 0 and n_levels > 1:
            sm = self.scale_max
            for l in range(n_levels - 1):
                p_l_down = F.max_pool2d(torch.sigmoid(position_maps[l]), 2)
                p_next = torch.sigmoid(position_maps[l + 1])
                # 用 GT 构造的风险标签 Q* 调节一致性强度（无梯度，防止网络
                # 抬高预测 Q 逃避一致性监督）：Q* 低 -> 目标应能安全通过
                # 下采样 -> 位置状态应一致；Q* 高允许变化
                w = F.max_pool2d(1.0 - risk_gt_levels[l], 2)
                stc_loss = stc_loss + (w * (p_l_down - p_next).abs()).mean()

                # 等效尺度演化：2 倍下采样后 log 域精确减半，
                # 再 max-pool 对齐下一层分辨率（与 mask 的 max-pool 口径一致）
                with torch.no_grad():
                    s_trans = torch.log1p(
                        torch.expm1(scale_maps[l].detach() * sm) / 2.0) / sm
                    s_trans = F.max_pool2d(s_trans, 2)
                    m_next = mask_levels[l + 1]
                    denom = m_next.sum() + 1e-6
                stc_loss = stc_loss + (
                    m_next * F.smooth_l1_loss(
                        scale_maps[l + 1], s_trans, reduction='none')
                ).sum() / denom
            stc_loss = stc_loss / (n_levels - 1)

        # Decoder Position Reconstruction：P_l^D 逐级对齐对应分辨率 GT。
        # 刻意不与 P_l^E 做任何直接一致性约束（保护 PCER 分歧—纠错机制）
        dec_pos_loss = target.new_zeros(())
        n_dec = 0
        if decoder_position_maps:
            H, W = target.shape[-2:]
            for p_d in decoder_position_maps:
                with torch.no_grad():
                    m_d = F.adaptive_max_pool2d(target, p_d.shape[-2:])
                    m_d = (m_d > 0.5).float()
                lvl_dec_loss = F.binary_cross_entropy_with_logits(p_d, m_d) \
                    + self._dice_loss(torch.sigmoid(p_d), m_d)
                # H/4 及以下的 decoder position 重建本身不稳定，降低权重
                if p_d.shape[-2] <= H // 4 or p_d.shape[-1] <= W // 4:
                    lvl_dec_loss = lvl_dec_loss * self.decoder_pos_top_weight
                dec_pos_loss = dec_pos_loss + lvl_dec_loss
                n_dec += 1
        if n_dec > 0:
            dec_pos_loss = dec_pos_loss / n_dec

        # 状态激活约束：高 Q 区域应该有更强的 FiLM 调制幅度，防止 gamma/beta 被绕成 0
        state_act_loss = target.new_zeros(())
        if self.state_activation_weight > 0 and neck_influence is not None:
            n_levels_inf = min(len(risk_maps), len(neck_influence))
            for l in range(n_levels_inf):
                q_prob = torch.sigmoid(risk_maps[l])
                infl = neck_influence[l]
                if q_prob.shape[-2:] != infl.shape[-2:]:
                    q_prob = F.interpolate(
                        q_prob, size=infl.shape[-2:],
                        mode='bilinear', align_corners=False)
                state_act_loss = state_act_loss + torch.exp(-(q_prob * infl)).mean()
            state_act_loss = state_act_loss / max(n_levels_inf, 1)

        # 证据竞争不再使用预定义 response map，无需 anchor 监督。
        # 第一层监督只保留：状态真实性（P/S/Q）+ decoder 位置链（P^D）。
        # survival loss 暂不加入 total（gate 已改为 Q-only）。
        total = self.pos_weight * pos_loss + self.scale_weight * scale_loss \
            + self.risk_weight * risk_loss \
            + self.state_consistency_weight * stc_loss \
            + self.decoder_pos_weight * dec_pos_loss \
            + self.state_activation_weight * state_act_loss
        parts = {
            'pos': pos_loss, 'scale': scale_loss, 'risk': risk_loss,
            'surv': surv_loss, 'stc': stc_loss, 'dec_pos': dec_pos_loss,
            'anchor': target.new_zeros(()), 'state_act': state_act_loss,
        }
        return total, parts


class AverageMeter(object):
    """Computes and stores the average and current value"""


    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
