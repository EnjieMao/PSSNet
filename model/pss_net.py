"""
model/pss_net.py
PSS-Net: Position-Scale-Survival Target State Dynamics Network.

核心观点
------------------------------------------------------------
P/S/Q 不是三个相互独立的先验，而是目标在第 l 个编码层级的状态：

    Z_l = (P_l, S_l, Q_l)

    P_l : 当前层级目标在哪里
    S_l : 当前层级目标的等效尺度，应以多大的空间范围观察目标及背景
    Q_l : 目标经历下一次分辨率压缩时发生证据稀释/消失的风险分数

当前版本（v6）采用状态驱动设计：
  - Encoder: P/S/Q 状态估计 + ADF（S 控制观测范围）+ Q-Router（Q 控制下采样）
  - Neck: State-conditioned FiLM（S/Q 调制多尺度特征）
  - Decoder: P-guided 目标/背景证据选择 + 位置状态链

主要 config
------------------------------------------------------------
base_channels(32) / use_se(True)
use_pss_heads(True)       关闭后退化为纯 U-Net baseline
use_scale_rf(True)        S 条件目标-背景观测几何
use_risk_downsample(True) 生存感知分辨率转换；关闭则用 MaxPool+Conv
q_gain_init(4.0)          Q 对 gate 的偏置强度
use_pos_skip(True)        decoder 跳跃连接的 P 增强
use_neck(True)            是否使用 FiLM Neck
neck_type('film')         'film' 轻量版

Forward
------------------------------------------------------------
masks, pred, state = model(x, tag=True)

state = {
    'position': [P0E, P1E, P2E],
    'scale': [S0, S1, S2],
    'risk': [Q0, Q1, Q2],
    'survival': [(uD0,uC0), ...],
    'gates': [G0,G1,G2],
    'decoder_position': [P0D,P1D,P2D,P3D],
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基础模块
# ============================================================


class SEResBlock(nn.Module):
    """
    Identity-preserving SE residual block.

    SE 只作用于 residual branch（相加之前完成调制），identity shortcut
    不被乘性抑制；in_ch == out_ch 时使用真 Identity：

        y = SiLU(x + SE(F(x)))
    """

    def __init__(self, in_ch, out_ch, reduction=8, use_se=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        if use_se:
            hidden = max(out_ch // reduction, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, hidden, kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.se = None

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = out * self.se(out)
        out = out + identity
        out = F.silu(out)
        return out


class PlainBlock(nn.Module):
    """
    最简非残差块：Conv-BN-SiLU × 2，无 shortcut、无 SE。
    用于消融实验验证残差连接的必要性。
    """

    def __init__(self, in_ch, out_ch, **_):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 2. Target State Estimator
# ============================================================


class PSSHead(nn.Module):
    """
    Target State Estimator：F_l (B,C,H,W) -> 状态 Z_l = (P, S, Q)

        P_l : position logits (B,1,H,W)
        S_l : normalized scale state，sigmoid -> (0,1)
        Q_l : downsampling survival-risk score，raw logits
    """

    def __init__(self, channels):
        super().__init__()
        h = max(channels // 4, 8)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.SiLU(inplace=True),
        )
        self.p_head = nn.Conv2d(h, 1, 1)
        self.s_head = nn.Conv2d(h, 1, 1)
        self.q_head = nn.Conv2d(h, 1, 1)

    def forward(self, x):
        h = self.shared(x)
        p = self.p_head(h)
        s = torch.sigmoid(self.s_head(h))
        q = self.q_head(h)  # raw logits; consumers apply sigmoid when needed
        return p, s, q


# ============================================================
# 3. Adaptive Deformable Field（S 条件神经感受野）
# ============================================================


class AdaptiveDeformableField(nn.Module):
    """
    自适应形变神经感受野（S 条件 Offset-Attention）。

    不再用固定公式把 S 映射为采样半径，而是由轻量调制器从 (F, S)
    直接学习每个像素的 3x3 采样偏移与调制权重：

        offsets = delta_max * tanh(h_off([F, S]))
        mod     = sigmoid(h_mod([F, S]))            （9 个逐点采样权重）

    S 只作为条件输入，采样几何完全由网络自适应学习。残差输出：

        F' = F + phi(Combine(mod_k * F(p + dp_k)))
    """

    def __init__(self, channels, delta_max=3.0):
        super().__init__()
        self.delta_max = delta_max
        h = max(channels // 4, 8)
        self.offset_head = nn.Sequential(
            nn.Conv2d(channels + 1, h, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, 18, 1),
        )
        # 零初始化：初始时无偏移，等价标准 3x3 卷积，训练稳定起步
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        self.mod_head = nn.Sequential(
            nn.Conv2d(channels + 1, h, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, 9, 1),
        )
        self.combine = nn.Sequential(
            nn.Conv2d(channels * 9, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, s):
        B, C, H, W = x.shape
        cond = torch.cat([x, s], dim=1)
        delta = self.delta_max * torch.tanh(self.offset_head(cond))  # (B,18,H,W)
        mod = torch.sigmoid(self.mod_head(cond))                     # (B,9,H,W)

        ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        sy = 2.0 / max(H - 1, 1)
        sx = 2.0 / max(W - 1, 1)

        taps = []
        idx = 0
        k = 0
        for ky in (-1.0, 0.0, 1.0):
            for kx in (-1.0, 0.0, 1.0):
                oy = ky + delta[:, idx]
                ox = kx + delta[:, idx + 1]
                idx += 2
                grid = torch.stack(
                    [gx.unsqueeze(0) + ox * sx, gy.unsqueeze(0) + oy * sy], dim=-1
                )
                taps.append(
                    F.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                                  align_corners=True) * mod[:, k: k + 1]
                )
                k += 1
        return x + F.silu(self.combine(torch.cat(taps, dim=1)))


# ============================================================
# 4. Survival-Aware Resolution Transition（Neural State Router）
# ============================================================


class BlurPool2d(nn.Module):
    """轻量模糊下采样：先 3x3 分组模糊，再 stride-2 平均池化。"""

    def __init__(self, channels):
        super().__init__()
        self.blur = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        # 固定为 3x3 均值模糊核，不学习
        self.blur.weight.data.fill_(1.0 / 9.0)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        with torch.no_grad():
            x = self.blur(x)
        return self.pool(x)


class NeuralStateRouter(nn.Module):
    """
    生存感知分辨率转换 / Detail-Context 分辨率融合（v7-fix3 + 对照模式）。

    候选路径：
        F_D = Conv(PixelUnshuffle(F))           detail，保留局部排列与高频细节
        F_C = LargeKernelConv(BlurPool(F))      context，强调全局背景/平滑上下文

    mode='q'（默认）：动态门控仅由 Q 驱动
        G = sigma( lambda_Q * (2*Q - 1) )
        out = G * F_D + (1 - G) * F_C

    mode='fixed'（对照）：去掉 Q，detail/context 固定 1:1 融合
        out = 0.5 * F_D + 0.5 * F_C

    mode='scalar'（对照）：全图共享一个可学习标量门 g=sigmoid(theta)，
        无空间自适应、无 Q 输入，用于隔离"空间自适应路由"的贡献
        out = g * F_D + (1 - g) * F_C
    """

    def __init__(self, in_ch, out_ch, q_gain_init=1.5, mode='q', map_hw=None):
        super().__init__()
        assert mode in ('q', 'fixed', 'scalar', 'map')
        self.mode = mode
        if mode == 'scalar':
            # theta=0 -> g=0.5 起步，与 fixed 对照同起点
            self.gate_logit = nn.Parameter(torch.zeros(1))
        elif mode == 'map':
            # 自由空间门（对照）：与门控同尺寸的可训练参数图，不从 Q/P 计算，
            # 用于分离"空间自适应性"与"状态条件化"的贡献；零初始化 -> 起步 0.5
            assert map_hw is not None
            self.gate_map = nn.Parameter(torch.zeros(1, 1, map_hw, map_hw))
        self.detail = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        # Context branch：BlurPool + 大核可分离卷积，强调背景上下文
        self.context = nn.Sequential(
            BlurPool2d(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        # 高风险区域额外增强 detail 分支的残差路径（零初始化，起步稳定）
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                      groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        nn.init.zeros_(self.detail_enhance[-2].weight)
        # 风险先验系数：通过 sigmoid 软约束到 (0, 2)，初始 q_gain_init=1.5
        import math
        if q_gain_init <= 0:
            raw_init = -2.0
        elif q_gain_init >= 2.0:
            raw_init = 2.0
        else:
            raw_init = math.log(q_gain_init / (2.0 - q_gain_init))
        self.q_gain_raw = nn.Parameter(torch.tensor(raw_init))
        # 共享生存评价器（保留输出供诊断，不参与 gate）
        hs = max(out_ch // 4, 8)
        self.surv_eval = nn.Sequential(
            nn.Conv2d(out_ch, hs, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hs, 1, 1),
        )

    def forward(self, x, q, s):
        f_d = self.detail(F.pixel_unshuffle(x, 2))
        f_c = self.context(x)

        if self.mode == 'fixed':
            # 无 Q 对照：detail / context 固定 1:1 融合
            gate = torch.full_like(f_d[:, :1], 0.5)
            out = 0.5 * f_d + 0.5 * f_c
        elif self.mode == 'scalar':
            # 可学习标量门控：全图共享 g，无空间自适应（对照实验）
            g = torch.sigmoid(self.gate_logit)
            gate = g.expand_as(f_d[:, :1])
            out = g * f_d + (1.0 - g) * f_c
        elif self.mode == 'map':
            # 自由空间门（对照）：可训练参数图上采样到门尺寸，逐像素凸组合，无 Q 输入
            gm = F.interpolate(self.gate_map, size=f_d.shape[-2:],
                               mode='bilinear', align_corners=False)
            gate = torch.sigmoid(gm)
            out = gate * f_d + (1.0 - gate) * f_c
        else:
            # q 为 logits，需要概率时先 sigmoid；max_pool 防止 tiny high-risk signal 被平均稀释
            q_half = F.max_pool2d(q, 2) if q is not None else torch.zeros_like(f_d[:, :1])
            q_prob = torch.sigmoid(q_half)

            # v7-fix3：纯 Q 驱动，无 feat_term，避免网络学负偏置压低 detail
            q_gain = 2.0 * torch.sigmoid(self.q_gain_raw)  # (0, 2)
            # q_prob=0.5 -> gate=0.5；q_prob=1 -> gate≈0.82（q_gain=1.5）
            q_term = q_gain * (2.0 * q_prob - 1.0)
            gate = torch.sigmoid(q_term)
            out = gate * f_d + (1.0 - gate) * f_c

            # 高风险目标额外增强 detail 证据，使 Q 对分辨率转换有不可绕过影响
            out = out + q_prob * F.silu(self.detail_enhance(f_d))

        # 反事实生存评估仅保留输出供诊断
        u_d = self.surv_eval(f_d)
        u_c = self.surv_eval(f_c)
        return (out, gate, (u_d, u_c))


class DetailPreservingTransition(nn.Module):
    """
    最深层固定 detail 保留的过渡模块。

    H/8 层 tiny target 的风险/位置状态不可靠，继续让 Q 做决策反而会把
    目标压没。因此该层固定使用 90% detail + 10% context 的固定融合策略，
    既保留高频证据，又提供少量背景偏置防止梯度异常。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            BlurPool2d(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        # 固定权重，不参与学习
        self.register_buffer('detail_weight', torch.tensor(0.9))

    def forward(self, x, q=None, s=None):
        f_d = self.detail(F.pixel_unshuffle(x, 2))
        f_c = self.context(x)
        out = self.detail_weight * f_d + (1.0 - self.detail_weight) * f_c
        gate = torch.ones_like(out[:, :1]) * self.detail_weight
        return out, gate, None


# ============================================================
# 5. Bottleneck Context
# ============================================================


class PlainDownsample(nn.Module):
    """普通下采样：MaxPool -> 3x3 Conv。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, q=None, s=None):
        return (self.down(x), None, None)


class ConvContextBlock(nn.Module):
    """Large-kernel depthwise context block。"""

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels, bias=False
        )
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return x + self.pw(self.dw(x))


# ============================================================
# 6. State-conditioned FiLM Neck
# ============================================================


class StateConditionedModulation(nn.Module):
    """
    Neck 子模块：State-conditioned Feature Modulation（FiLM）。

    用 S/Q 状态对 encoder 特征做 FiLM 式调制：
        \hat F_l = \gamma(Z_l) \odot F_l + \beta(Z_l),
        Z_l = [S_l, Q_l]

    注意：Neck 不负责定位，因此 FiLM 只输入 S/Q（尺度与风险），
    不使用 P（位置）。P 仅由 Encoder 估计、Decoder 恢复。
    """

    def __init__(self, c0, c1, c2, c3, proj_ch):
        super().__init__()
        # 特征投影
        self.feat0 = nn.Conv2d(c0, proj_ch, 1, bias=False)
        self.feat1 = nn.Conv2d(c1, proj_ch, 1, bias=False)
        self.feat2 = nn.Conv2d(c2, proj_ch, 1, bias=False)
        self.feat3 = nn.Conv2d(c3, proj_ch, 1, bias=False)

        # 由 S/Q 生成 gamma/beta（2 * proj_ch）
        h = max(proj_ch // 4, 8)
        self.film0 = nn.Sequential(
            nn.Conv2d(2, h, 1, bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(h, proj_ch * 2, 1),
        )
        self.film1 = nn.Sequential(
            nn.Conv2d(2, h, 1, bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(h, proj_ch * 2, 1),
        )
        self.film2 = nn.Sequential(
            nn.Conv2d(2, h, 1, bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(h, proj_ch * 2, 1),
        )
        self.film3 = nn.Sequential(
            nn.Conv2d(2, h, 1, bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(h, proj_ch * 2, 1),
        )

    @staticmethod
    def _sq(s, q, target_size):
        if s is None or q is None:
            return None
        if s.shape[-2:] != target_size:
            s = F.interpolate(s, size=target_size, mode='bilinear', align_corners=False)
        # q 为 logits，FiLM 条件需要概率语义
        if q.shape[-2:] != target_size:
            q = F.interpolate(q, size=target_size, mode='bilinear', align_corners=False)
        q = torch.sigmoid(q)
        return torch.cat([s, q], dim=1)

    def forward(self, f0, f1, f2, f3, s0, s1, s2, q0, q1, q2):
        B, _, H3, W3 = f3.shape
        device, dtype = f3.device, f3.dtype

        def zero_sq(size):
            return torch.zeros(B, 2, size[0], size[1], device=device, dtype=dtype)

        sq0 = self._sq(s0, q0, f0.shape[-2:]) if s0 is not None else zero_sq(f0.shape[-2:])
        sq1 = self._sq(s1, q1, f1.shape[-2:]) if s1 is not None else zero_sq(f1.shape[-2:])
        sq2 = self._sq(s2, q2, f2.shape[-2:]) if s2 is not None else zero_sq(f2.shape[-2:])
        sq3 = self._sq(s2, q2, (H3, W3)) if s2 is not None else zero_sq((H3, W3))

        x0 = self.feat0(f0)
        x1 = self.feat1(f1)
        x2 = self.feat2(f2)
        x3 = self.feat3(f3)

        g0, b0 = self.film0(sq0).chunk(2, dim=1)
        g1, b1 = self.film1(sq1).chunk(2, dim=1)
        g2, b2 = self.film2(sq2).chunk(2, dim=1)
        g3, b3 = self.film3(sq3).chunk(2, dim=1)

        y0 = x0 * (1.0 + g0) + b0
        y1 = x1 * (1.0 + g1) + b1
        y2 = x2 * (1.0 + g2) + b2
        y3 = x3 * (1.0 + g3) + b3

        # 记录每层 FiLM 的调制幅度，用于状态激活约束（防止 gamma/beta 被绕成 0）
        infl0 = (g0.abs().mean(dim=1, keepdim=True) + b0.abs().mean(dim=1, keepdim=True))
        infl1 = (g1.abs().mean(dim=1, keepdim=True) + b1.abs().mean(dim=1, keepdim=True))
        infl2 = (g2.abs().mean(dim=1, keepdim=True) + b2.abs().mean(dim=1, keepdim=True))
        infl3 = (g3.abs().mean(dim=1, keepdim=True) + b3.abs().mean(dim=1, keepdim=True))

        return (y0, y1, y2, y3), (infl0, infl1, infl2, infl3)


class StateConditionedNeck(nn.Module):
    """
    轻量 State-conditioned FiLM Neck。

    只做 S/Q 状态对每个 encoder 尺度的 FiLM 调制，不做跨尺度注意力
    或 survival 补偿。目标是让 Neck 成为"状态驱动的特征调制器"，同时
    避免复杂交互带来的优化困难。

    流程：
        f_l → Conv1x1 → FiLM(S_l, Q_l) → Conv1x1 → n_l
    """

    def __init__(self, c0=32, c1=64, c2=128, c3=256, proj_ch=32):
        super().__init__()
        self.modulation = StateConditionedModulation(c0, c1, c2, c3, proj_ch)
        self.out0 = nn.Conv2d(proj_ch, c0, 1, bias=False)
        self.out1 = nn.Conv2d(proj_ch, c1, 1, bias=False)
        self.out2 = nn.Conv2d(proj_ch, c2, 1, bias=False)
        self.out3 = nn.Conv2d(proj_ch, c3, 1, bias=False)

    def forward(self, f0, f1, f2, f3, s0, s1, s2, q0, q1, q2):
        (m0, m1, m2, m3), (i0, i1, i2, i3) = self.modulation(
            f0, f1, f2, f3, s0, s1, s2, q0, q1, q2)
        return (
            (self.out0(m0), self.out1(m1), self.out2(m2), self.out3(m3)),
            (i0, i1, i2, i3),
        )


# ============================================================
# 6b. Frequency Decomposition Neck
# ============================================================


class FrequencyDecompositionNeck(nn.Module):
    """
    频域解耦 + 双向跨尺度融合 Neck。

    设计哲学：
      - Encoder 负责"怎么在下采样时保留好"
      - Neck 负责"多尺度增强融合"

    两步走：
      1. 频域解耦增强：每尺度显式分解为高频（目标细节）和低频（背景），
         分别增强。
      2. 双向门控融合：高层语义下传 + 低层细节上送，内容自适应门控。

    兼容旧接口：多余的 s/q 参数会被 *args, **kwargs 吸收并忽略。
    """

    def __init__(self, c0, c1, c2, c3, proj_ch=64):
        super().__init__()
        channels = [c0, c1, c2, c3]
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, proj_ch, 1, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.SiLU(inplace=True),
            ) for c in channels
        ])

        # 低频提取：AvgPool + 大核 depthwise
        self.low_pass = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                nn.Conv2d(proj_ch, proj_ch, 5, padding=2,
                          groups=proj_ch, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.SiLU(inplace=True),
            ) for _ in range(4)
        ])

        # 高频增强门控，初始偏置为正，起步时不抑制高频
        self.high_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch, proj_ch // 4, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(proj_ch // 4, proj_ch, 1, bias=True),
                nn.Sigmoid(),
            ) for _ in range(4)
        ])
        for hg in self.high_gate:
            nn.init.zeros_(hg[-2].weight)
            nn.init.constant_(hg[-2].bias, 1.0)

        # 低频增强：背景上下文聚合
        self.low_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch, proj_ch, 7, padding=3,
                          groups=proj_ch, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(proj_ch, proj_ch, 1, bias=False),
                nn.BatchNorm2d(proj_ch),
            ) for _ in range(4)
        ])

        # 频域融合
        self.freq_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch * 2, proj_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.SiLU(inplace=True),
            ) for _ in range(4)
        ])

        # 双向跨尺度融合
        self.td_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch, proj_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.SiLU(inplace=True),
            ) for _ in range(3)
        ])
        self.bu_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch, proj_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.SiLU(inplace=True),
            ) for _ in range(3)
        ])

        # 跨尺度门控：3 td + 3 bu
        self.cross_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch * 2, proj_ch // 4, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(proj_ch // 4, 1, 1, bias=True),
                nn.Sigmoid(),
            ) for _ in range(6)
        ])
        # 零初始化最后一层，sigmoid(0)=0.5，起步时不偏向注入或保留
        for cg in self.cross_gate:
            nn.init.zeros_(cg[-2].weight)
            nn.init.zeros_(cg[-2].bias)

        # 精修
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch, proj_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.SiLU(inplace=True),
            ) for _ in range(4)
        ])

        self.out = nn.ModuleList([
            nn.Conv2d(proj_ch, c, 1, bias=False) for c in channels
        ])

    def forward(self, f0, f1, f2, f3, *args, **kwargs):
        feats = [f0, f1, f2, f3]
        x = [self.proj[i](feats[i]) for i in range(4)]

        # Step 1: 频域解耦增强
        decomp = []
        for i in range(4):
            l = self.low_pass[i](x[i])
            h = x[i] - l
            h = h * self.high_gate[i](h)
            l = l + self.low_enhance[i](l)
            y = self.freq_fuse[i](torch.cat([h, l], dim=1))
            decomp.append(y)

        # Step 2: 自顶向下
        td = decomp.copy()
        g_idx = 0
        for i in reversed(range(3)):
            up = F.interpolate(
                td[i + 1], size=td[i].shape[-2:],
                mode='bilinear', align_corners=False)
            up = self.td_conv[i](up)
            gate = self.cross_gate[g_idx](torch.cat([td[i], up], dim=1))
            td[i] = td[i] + gate * up
            g_idx += 1

        # Step 3: 自底向上
        bu = td.copy()
        for i in range(3):
            down = self.bu_conv[i](bu[i])
            if down.shape[-2:] != bu[i + 1].shape[-2:]:
                down = F.interpolate(
                    down, size=bu[i + 1].shape[-2:],
                    mode='bilinear', align_corners=False)
            gate = self.cross_gate[g_idx](torch.cat([bu[i + 1], down], dim=1))
            bu[i + 1] = bu[i + 1] + gate * down
            g_idx += 1

        # Step 4: 精修 + 恢复通道
        outs = []
        for i in range(4):
            r = self.refine[i](bu[i])
            outs.append(self.out[i](r))

        return tuple(outs), None


# ============================================================
# 7. P-guided Decoder
# ============================================================


class DecoderPositionHead(nn.Module):
    """Decoder reconstruction position state estimator，输出 logits P_l^D。"""

    def __init__(self, channels):
        super().__init__()
        h = max(channels // 4, 8)
        self.head = nn.Sequential(
            nn.Conv2d(channels, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, 1, 1),
        )

    def forward(self, x):
        return self.head(x)


class PSSDecoderBlock(nn.Module):
    """
    P-guided Decoder Block：P 状态直接选择 encoder skip 的目标/背景证据。

    相比普通 U-Net decoder，增加两个轻量分支：
      - target_branch: 处理目标区域的 skip 特征
      - context_branch: 处理背景区域的 skip 特征
    由 P 状态（sigmoid 后）决定空间比例：
        skip_sel = P * target_branch(skip) + (1 - P) * context_branch(skip)

    这保留了 PSS 设计中 "P 控制空间证据选择" 的核心思想。

    v7-fix3：恢复 fix1 的完整 P-gating， Encoder P 只作为状态估计器，
    在 decoder 中 detach 使用，防止 P head 退化为"分支选择器"。
    gate_alpha/gate_beta：门控界 G_P = alpha + beta*sigmoid(P)，默认 0.1/0.8。
    """

    def __init__(self, c_high, c_low, c_out, gate_alpha=0.1, gate_beta=0.8,
                 gate_mode='p'):
        super().__init__()
        self.gate_alpha = gate_alpha
        self.gate_beta = gate_beta
        self.gate_mode = gate_mode
        if gate_mode == 'scalar':
            # theta=0 -> g=0.5 起步，与 p=None 的固定融合对照同起点
            self.gate_logit = nn.Parameter(torch.zeros(1))
        self.proj = nn.Sequential(
            nn.Conv2d(c_high, c_low, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.SiLU(inplace=True),
        )
        # v7: target_branch 强调目标高频/局部细节
        self.target_branch = nn.Sequential(
            nn.Conv2d(c_low, c_low, kernel_size=3, padding=2,
                      dilation=2, groups=c_low, bias=False),
            nn.BatchNorm2d(c_low),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_low, c_low, kernel_size=1, bias=False),
        )
        # v7: context_branch 强调背景平滑/低频上下文
        self.context_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_low, c_low, kernel_size=7, padding=3,
                      groups=c_low, bias=False),
            nn.BatchNorm2d(c_low),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_low, c_low, kernel_size=1, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_low * 2, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        self.pos_head = DecoderPositionHead(c_out)

    def forward(self, x_high, x_low, p):
        y = F.interpolate(
            x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=False
        )
        y = self.proj(y)
        skip_t = self.target_branch(x_low)
        skip_c = self.context_branch(x_low)

        # Encoder P 只作为状态估计器使用，不参与 Decoder 反向传播，
        # 防止 P head 退化为“分支选择器”。
        if self.gate_mode == 'scalar':
            # 可学习标量门控（对照）：全图共享 g，无空间自适应、忽略 P
            g = torch.sigmoid(self.gate_logit)
            p_prob = g.expand_as(y[:, :1])
        elif p is not None:
            p_prob = torch.sigmoid(p.detach())
            # 不对称保底：P=0 时仍保留 gate_alpha 的 target 分支，
            # P=1 时保留 (1-gate_alpha-gate_beta) 的 context。
            p_prob = self.gate_alpha + self.gate_beta * p_prob
        else:
            # baseline / ablation 配置：p=None（use_pss_heads=False）
            p_prob = torch.full_like(y[:, :1], 0.5)
        skip_sel = p_prob * skip_t + (1.0 - p_prob) * skip_c

        out = self.fuse(torch.cat([skip_sel, y], dim=1))
        p_out = self.pos_head(out)
        return out, p_out


class SimpleDecoderBlock(nn.Module):
    """
    最简 U-Net decoder block：上采样 + 拼接 + 卷积。
    关闭 PSS/PCER decoder 时使用的对照结构。

        y = Upsample(x_high)
        out = Conv3x3(concat([y, x_low]))
    """

    def __init__(self, c_high, c_low, c_out):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c_high, c_low, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_low * 2, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x_high, x_low, p=None):
        y = F.interpolate(
            x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=False
        )
        y = self.proj(y)
        out = self.fuse(torch.cat([x_low, y], dim=1))
        return out, None


class ConcatCBSBlock(nn.Module):
    """
    三输入 channel-match + Concat + CBS Decoder Block。

    把三个来源的特征统一匹配到 c_low 通道后直接拼接，再用
    Conv-BN-SiLU（CBS）融合，不做 target/context 双分支与 P 门控：

        y   = Conv1x1(Up(x_high))        -> c_low
        p_f = Conv1x1(sigmoid(p.detach()))-> c_low   （无 P 时为 0）
        out = CBS(concat([y, x_low, p_f]))

    与 PSSDecoderBlock 的区别：P 不再做软门控选择分支，而是作为
    一路独立证据与高层语义、skip 细节并列 concat，由 CBS 自行学习融合。
    Encoder P 同样 detach，保持“状态估计器”语义。
    """

    def __init__(self, c_high, c_low, c_out):
        super().__init__()
        self.proj_high = nn.Sequential(
            nn.Conv2d(c_high, c_low, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.SiLU(inplace=True),
        )
        self.proj_p = nn.Conv2d(1, c_low, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(c_low * 3, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x_high, x_low, p=None):
        y = F.interpolate(
            x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=False
        )
        y = self.proj_high(y)
        if p is not None:
            p_feat = self.proj_p(torch.sigmoid(p.detach()))
        else:
            p_feat = torch.zeros_like(y)
        out = self.fuse(torch.cat([y, x_low, p_feat], dim=1))
        return out, None


# ============================================================
# 8. Complete PSS-Net
# ============================================================


class PSSNet(nn.Module):
    """PSS-Net with P-guided Decoder and FiLM Neck。"""

    def __init__(self, size, input_channels, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.config = config

        self.base_channels = config.get('base_channels', 32)
        self.use_se = config.get('use_se', True)
        self.use_resblock = config.get('use_resblock', True)
        self.use_pss_heads = config.get('use_pss_heads', True)
        self.use_scale_rf = config.get('use_scale_rf', True)
        self.use_risk_downsample = config.get('use_risk_downsample', True)
        self.risk_downsample_mode = config.get('risk_downsample_mode', 'q')
        self.pcer_gate_mode = config.get('pcer_gate_mode', 'p')
        self.use_pos_skip = config.get('use_pos_skip', True)
        self.use_neck = config.get('use_neck', True)
        self.neck_type = config.get('neck_type', 'film')
        self.neck_proj_ch = config.get('neck_proj_ch', 64)
        self.q_gain_init = config.get('q_gain_init', 4.0)
        self.use_deep_supervision = config.get('use_deep_supervision', True)
        # use_context_block：瓶颈处的大核深度可分离上下文模块（ConvContextBlock）
        self.use_context_block = config.get('use_context_block', True)
        # use_pss_decoder=True：使用 P-guided PCER decoder；
        # False：退化为最简上采样+拼接+卷积的 U-Net decoder
        self.use_pss_decoder = config.get('use_pss_decoder', True)
        # decoder_type：'pss'（默认，等同 use_pss_decoder=True）、'simple'、
        # 'concat_cbs'（三输入 channel-match + concat + CBS，P 作为独立证据路）
        self.decoder_type = config.get('decoder_type', None)

        c1 = self.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        # Stem：stride 1，保住全分辨率细节
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # Encoder: stage 0/1/2 + stride-8 bottleneck
        EncBlock = SEResBlock if self.use_resblock else PlainBlock
        self.enc0 = EncBlock(c1, c1, use_se=self.use_se)
        self.enc1 = EncBlock(c2, c2, use_se=self.use_se)
        self.enc2 = EncBlock(c3, c3, use_se=self.use_se)

        if self.use_pss_heads:
            self.head0 = PSSHead(c1)
            self.head1 = PSSHead(c2)
            self.head2 = PSSHead(c3)

        self.rf0 = AdaptiveDeformableField(c1) if self.use_scale_rf else None
        self.rf1 = AdaptiveDeformableField(c2) if self.use_scale_rf else None
        self.rf2 = AdaptiveDeformableField(c3) if self.use_scale_rf else None

        # down0/down1/down2 全部使用 Q 控制的动态 Router
        # 自由空间门（mode='map'）需要各层门图尺寸：size/2, size/4, size/8
        rds = self.risk_downsample_mode
        mh = [size // 2, size // 4, size // 8] if rds == 'map' else [None] * 3
        self.down0 = NeuralStateRouter(
            c1, c2, q_gain_init=self.q_gain_init, mode=rds, map_hw=mh[0]
        ) if self.use_risk_downsample else PlainDownsample(c1, c2)
        self.down1 = NeuralStateRouter(
            c2, c3, q_gain_init=self.q_gain_init, mode=rds, map_hw=mh[1]
        ) if self.use_risk_downsample else PlainDownsample(c2, c3)
        self.down2 = NeuralStateRouter(
            c3, c4, q_gain_init=self.q_gain_init, mode=rds, map_hw=mh[2]
        ) if self.use_risk_downsample else PlainDownsample(c3, c4)

        self.enc3 = EncBlock(c4, c4, use_se=self.use_se)
        self.context = ConvContextBlock(c4) if self.use_context_block else nn.Identity()

        # Decoder：根据配置选择 PSS/PCER decoder、最简 U-Net decoder
        # 或三输入 concat + CBS decoder
        if self.decoder_type == 'concat_cbs':
            DecoderBlock = ConcatCBSBlock
            self.dec_pos3 = None
        elif self.use_pss_decoder:
            DecoderBlock = lambda c_h, c_l, c_o: PSSDecoderBlock(
                c_h, c_l, c_o,
                gate_alpha=config.get('pcer_gate_alpha', 0.1),
                gate_beta=config.get('pcer_gate_beta', 0.8),
                gate_mode=config.get('pcer_gate_mode', 'p'))
            self.dec_pos3 = DecoderPositionHead(c4)
        else:
            DecoderBlock = SimpleDecoderBlock
            self.dec_pos3 = None
        self.dec2 = DecoderBlock(c4, c3, c3)
        self.dec1 = DecoderBlock(c3, c2, c2)
        self.dec0 = DecoderBlock(c2, c1, c1)

        # Neck：状态引导多尺度证据融合（film）或频域解耦融合（freq）
        if self.use_neck:
            if self.neck_type == 'freq':
                self.neck = FrequencyDecompositionNeck(
                    c0=c1, c1=c2, c2=c3, c3=c4, proj_ch=self.neck_proj_ch)
            else:
                self.neck = StateConditionedNeck(
                    c0=c1, c1=c2, c2=c3, c3=c4, proj_ch=self.neck_proj_ch)
        else:
            self.neck = None

        # 最终预测头
        self.seg_head = nn.Conv2d(c1, 1, 1)

        # 深监督辅助头（仅训练 tag=True 时使用），输出分辨率 H/2 与 H/4
        if self.use_deep_supervision:
            self.output_1 = nn.Conv2d(c2, 1, 1)
            self.output_2 = nn.Conv2d(c3, 1, 1)

    def _stage(self, x, enc, head, rf, down):
        """
        Encoder stage：feature encoding -> state estimation ->
        scale-conditioned observation -> survival-aware transition。
        P/S/Q 不再 detach，允许下游模块（ADF / Router / Decoder / Neck）
        的梯度反馈到状态头，使状态从"描述变量"变成"决策变量"。
        """
        f = enc(x)
        if self.use_pss_heads:
            p, s, q = head(f)
        else:
            p = None
            s = None
            q = None
        if rf is not None:
            if s is None:
                s = torch.zeros_like(f[:, :1])
            f = rf(f, s)
        f_down, gate, surv = down(
            f,
            q if q is not None else None,
            s if s is not None else None,
        )
        return (f, p, s, q, f_down, gate, surv)

    def forward(self, x, tag=False):
        # Encoder
        f0, p0, s0, q0, d0, gt0, v0 = self._stage(
            self.stem(x), self.enc0, getattr(self, 'head0', None), self.rf0, self.down0
        )
        f1, p1, s1, q1, d1, gt1, v1 = self._stage(
            d0, self.enc1, getattr(self, 'head1', None), self.rf1, self.down1
        )
        f2, p2, s2, q2, d2, gt2, v2 = self._stage(
            d1, self.enc2, getattr(self, 'head2', None), self.rf2, self.down2
        )
        f3 = self.context(self.enc3(d2))

        # Neck：状态引导多尺度证据融合（仅使用 S/Q 状态，P 留给 Decoder）
        if self.use_neck and self.neck is not None:
            (n0, n1, n2, n3), neck_influence = self.neck(
                f0, f1, f2, f3, s0, s1, s2, q0, q1, q2)
        else:
            # baseline：退化为纯 U-Net，Neck 恒等
            n0, n1, n2, n3 = f0, f1, f2, f3
            neck_influence = None

        # Decoder
        if self.decoder_type == 'concat_cbs':
            # 三输入 concat + CBS：P 作为独立证据路直接参与拼接
            x_d2, _ = self.dec2(n3, n2, p2)
            x_d1, _ = self.dec1(x_d2, n1, p1)
            x_d0, _ = self.dec0(x_d1, n0, p0)
            decoder_positions = None
        elif self.use_pss_decoder:
            pd3 = self.dec_pos3(n3)
            # pcer_gate_mode != 'p' 时切断 P 对选择门控的作用（fixed/scalar 对照），
            # P 状态头及其监督保留，仅解除条件化
            dp = self.pcer_gate_mode == 'p'
            x_d2, pd2 = self.dec2(n3, n2, p2 if dp else None)
            x_d1, pd1 = self.dec1(x_d2, n1, p1 if dp else None)
            x_d0, pd0 = self.dec0(x_d1, n0, p0 if dp else None)
            decoder_positions = [pd0, pd1, pd2, pd3]
        else:
            x_d2, _ = self.dec2(n3, n2)
            x_d1, _ = self.dec1(x_d2, n1)
            x_d0, _ = self.dec0(x_d1, n0)
            decoder_positions = None

        # 最终预测
        mask = self.seg_head(x_d0)

        if self.use_pss_heads:
            position_maps = [p0, p1, p2]
            scale_maps = [s0, s1, s2]
            risk_maps = [q0, q1, q2]
        else:
            position_maps = None
            scale_maps = None
            risk_maps = None

        surv_pairs = [v for v in (v0, v1, v2) if v is not None]
        gates = [g for g in (gt0, gt1, gt2) if g is not None]

        state = {
            'position': position_maps,
            'scale': scale_maps,
            'risk': risk_maps,
            'survival': surv_pairs if surv_pairs else None,
            'gates': gates if gates else None,
            'decoder_position': decoder_positions,
            'neck_influence': neck_influence,
        }

        if tag:
            if self.use_deep_supervision:
                mask1 = self.output_1(x_d1)
                mask2 = self.output_2(x_d2)
                return ([mask, mask1, mask2], mask, state)
            return ([], mask, state)
        return ([], mask, state)


# ============================================================
# 9. Sanity Test
# ============================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 1, 256, 256, device=device)

    configs = {
        'full': {},
        'no_neck': {'use_neck': False},
        'baseline': {
            'use_pss_heads': False,
            'use_scale_rf': False,
            'use_risk_downsample': False,
            'use_pos_skip': False,
            'use_neck': False,
        },
    }

    for name, cfg in configs.items():
        print('\n===== %s =====' % name)
        model = PSSNet(size=256, input_channels=1, config=cfg).to(device)
        masks, pred, state = model(x, tag=True)
        assert pred.shape == (2, 1, 256, 256), pred.shape
        print('prediction:', pred.shape)
        print('deep-supervision masks:', [m.shape for m in masks])
        if state['position'] is not None:
            print('encoder P levels:', [tuple(p.shape[-2:]) for p in state['position']])
            print('decoder P levels:', [tuple(p.shape[-2:]) for p in state['decoder_position']])
        if state['gates']:
            print('gate levels:', [tuple(g.shape[-2:]) for g in state['gates']])
        loss = pred.sigmoid().mean()
        loss.backward()
        params = sum(p.numel() for p in model.parameters())
        print('params: %.2f M' % (params / 1e6))

    print('\nOK')
