from __future__ import absolute_import, print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class GatheringLoss(nn.Module):
    def __init__(self, reduction='none', memto_framework=True):
        super(GatheringLoss, self).__init__()
        self.reduction = reduction
        self.memto_framework = memto_framework

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        score = torch.matmul(query, key.T)  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1)  # TxM
        return score
    
    def forward(self, queries, items):
        '''
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)

        loss_mse = torch.nn.MSELoss(reduction=self.reduction)

        #  To eliminate the impact of magnitude, we use the queries in the unit magnitude
        f = torch.fft.rfft(queries, dim=-2).permute(0, 2, 1)
        i_query_angle = torch.angle(f)
        unit_magnitude_queries = torch.fft.irfft(torch.exp(-1j * i_query_angle)).permute(0, 2, 1)

        if self.memto_framework:
            score = torch.einsum('bij,kj->bik', unit_magnitude_queries, items)
            # score = torch.einsum('bij,kj->bik', queries, items)
            _, indices = torch.topk(score, 1, dim=-1)
            step_basis = torch.gather(items.unsqueeze(0).repeat(batch_size, 1, 1), 1, indices.expand(-1, -1, items.size(-1)))
            gathering_loss = loss_mse(queries, step_basis)

        else:
            score = torch.einsum('bij,bkj->bik', unit_magnitude_queries, items)
            # score = torch.einsum('bij,bkj->bik', queries, items)
            _, indices = torch.topk(score, 1, dim=-1)
            C = torch.gather(items, 1, indices.expand(-1, -1, items.size(-1)))
            gathering_loss = loss_mse(queries, C)

        if not self.reduction == 'none':
            return gathering_loss
        
        gathering_loss = torch.sum(gathering_loss, dim=-1)  # T
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)   # N x L

        return gathering_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        '''
        x (attn_weights) : TxM
        '''
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss


class SmoothEpsilonTubeLoss(nn.Module):
    """
    借鉴 SVR 的 ε-insensitive 思想的可微分损失函数
    
    核心思想：
      - 管道内（误差 < ε）：小惩罚（二次损失，容忍正常波动/噪声）
      - 管道外（误差 > ε）：大惩罚（线性损失，聚焦异常偏差）
    
    效果：
      - 训练时：自动过滤小的正常波动，让模型聚焦学习显著模式
      - 测试时：管道外的点更可能是异常
    
    公式：
      L(e) = e²/(2ε)      if |e| ≤ ε  (管道内)
      L(e) = |e| - ε/2    if |e| > ε  (管道外)
    
    这是 Huber Loss 的变体，ε 控制管道宽度
    """
    def __init__(self, epsilon=0.1, learnable=True, adaptive_mode='none'):
        super().__init__()
        self.adaptive_mode = adaptive_mode
        
        if adaptive_mode == 'none':
            # 固定 ε 或简单可学习 ε
            if learnable:
                self.epsilon = nn.Parameter(torch.tensor(epsilon))
            else:
                self.register_buffer('epsilon', torch.tensor(epsilon))
        elif adaptive_mode == 'feature_based':
            # 基于特征的自适应 ε
            self.epsilon_base = nn.Parameter(torch.tensor(epsilon))
            self.epsilon_net = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        elif adaptive_mode == 'error_based':
            # 基于误差分布的自适应 ε
            self.epsilon_base = nn.Parameter(torch.tensor(epsilon))
            self.alpha = nn.Parameter(torch.tensor(0.1))  # 自适应系数
            self.register_buffer('error_ema', torch.tensor(epsilon))  # 误差的指数移动平均
            self.momentum = 0.99
        
        self.learnable = learnable
    
    def forward(self, x_hat, x):
        """
        x_hat: (B, T, C) 重构输出
        x:     (B, T, C) 原始输入
        
        Returns:
            loss: 标量损失值
        """
        # 逐点误差
        error = torch.abs(x_hat - x)  # (B, T, C)
        
        # 自适应 ε 计算
        if self.adaptive_mode == 'none':
            eps = torch.abs(self.epsilon) + 1e-8
        elif self.adaptive_mode == 'feature_based':
            # 基于当前误差幅度自适应调整 ε
            error_magnitude = error.mean().unsqueeze(0)  # (1,)
            eps_scale = self.epsilon_net(error_magnitude)  # (1,)
            eps = torch.abs(self.epsilon_base) * (0.1 + 1.9 * eps_scale) + 1e-8  # 0.1x ~ 2x 范围
        elif self.adaptive_mode == 'error_based':
            # 基于误差分布的指数移动平均自适应 ε
            current_error_mean = error.mean().detach()
            if self.training:
                # 更新误差的指数移动平均
                self.error_ema = self.momentum * self.error_ema + (1 - self.momentum) * current_error_mean
            
            # ε = base_ε + α * error_ema
            eps = torch.abs(self.epsilon_base) + torch.abs(self.alpha) * self.error_ema + 1e-8
        
        # 管道内外判断
        inside_mask = (error <= eps).float()
        outside_mask = 1.0 - inside_mask
        
        # 平滑 ε-tube 损失
        # 管道内：二次损失（小惩罚，容忍噪声）
        # 管道外：线性损失（大惩罚，聚焦异常）
        loss_inside = error ** 2 / (2 * eps)
        loss_outside = error - eps / 2
        
        loss = inside_mask * loss_inside + outside_mask * loss_outside
        
        return loss.mean()
    
    def get_epsilon(self):
        """返回当前的 ε 值（用于监控）"""
        if self.adaptive_mode == 'none':
            return torch.abs(self.epsilon).item()
        elif self.adaptive_mode == 'feature_based':
            return torch.abs(self.epsilon_base).item()
        elif self.adaptive_mode == 'error_based':
            return (torch.abs(self.epsilon_base) + torch.abs(self.alpha) * self.error_ema).item()
        
    def get_adaptive_info(self):
        """返回自适应 ε 的详细信息"""
        info = {'mode': self.adaptive_mode}
        if self.adaptive_mode == 'feature_based':
            info['epsilon_base'] = torch.abs(self.epsilon_base).item()
        elif self.adaptive_mode == 'error_based':
            info['epsilon_base'] = torch.abs(self.epsilon_base).item()
            info['alpha'] = torch.abs(self.alpha).item()
            info['error_ema'] = self.error_ema.item()
            info['current_epsilon'] = self.get_epsilon()
        return info


class ProtoSVDDLoss(nn.Module):
    """
    最终落地版 不对称ProtoSVDD损失
    
    核心思想：
      - 使用永久固定的正弦原型库作为黄金参考
      - 频域投影头轻量对齐，时域投影头对齐到原型
      - 时域和频域必须选择同一个原型（分配一致性）
    
    所有超参数已经经过调优，不需要修改
    """
    def __init__(self, nu=0.05, R=2.0, lambda_consist=0.2, temperature=0.15, R_min=0.1):
        super().__init__()
        self.nu = nu
        self.R_min = R_min
        self.log_R = nn.Parameter(torch.log(torch.tensor(R)))
        self.lambda_consist = lambda_consist
        self.temperature = temperature
        self.last_metrics = {}

    @property
    def R_sq(self):
        return torch.exp(torch.clamp(self.log_R, min=math.log(self.R_min))) ** 2

    def forward(self, z_time, z_freq, mem_R, update_R=True):
        """
        Args:
            z_time: [B, T, D] 时域SVDD投影特征
            z_freq: [B, T, D] 频域SVDD投影特征
            mem_R: 原来的正弦原型库，原封不动传进来
        """
        B, T, D = z_time.shape

        # ============================================
        # 🧊 黄金原型库 永久固定 永远不可训练
        # ============================================
        P = F.normalize(mem_R.T, p=2, dim=-1)
        P = P.to(z_time.device)

        # 归一化特征
        zt = F.normalize(z_time.reshape(-1, D), p=2, dim=-1)
        zf = F.normalize(z_freq.reshape(-1, D), p=2, dim=-1)

        # ============================================
        # 到原型的距离
        # ============================================
        dist_t = 2 - 2 * zt @ P.T  # 余弦距离，等价于L2距离平方
        dist_f = 2 - 2 * zf @ P.T

        min_dist_t, _ = dist_t.min(dim=-1)
        min_dist_f, _ = dist_f.min(dim=-1)

        # ============================================
        # SVDD 紧凑性损失
        # ============================================
        loss_compact_t = torch.mean(F.relu(min_dist_t - self.R_sq))
        loss_compact_f = torch.mean(F.relu(min_dist_f - self.R_sq))
        loss_compact = (loss_compact_t + loss_compact_f) / self.nu

        # ============================================
        # 原型分配一致性损失 最有价值的新增项
        # 时域和频域必须选择同一个原型
        # ============================================
        assign_t = F.softmax(-dist_t / self.temperature, dim=-1)
        assign_f = F.softmax(-dist_f / self.temperature, dim=-1)

        loss_assign = 1 - torch.mean(torch.sum(assign_t * assign_f, dim=-1))

        # ============================================
        # 半径正则
        # ============================================
        loss_R = self.R_sq * 0.01

        # ============================================
        # 动态更新半径
        # ============================================
        if update_R:
            with torch.no_grad():
                all_dist = torch.cat([min_dist_t, min_dist_f])
                new_R = torch.quantile(all_dist, 1 - self.nu)
                new_R = torch.clamp(new_R, min=self.R_min)
                self.log_R.data = 0.95 * self.log_R.data + 0.05 * torch.log(new_R + 1e-6)
                self.log_R.data = torch.clamp(self.log_R.data, min=math.log(self.R_min))

        # ============================================
        # 总损失
        # ============================================
        total_loss = loss_compact + self.lambda_consist * loss_assign + loss_R

        self.last_metrics = {
            'compact_t': loss_compact_t.item(),
            'compact_f': loss_compact_f.item(),
            'assign': loss_assign.item(),
            'R': torch.sqrt(self.R_sq).item()
        }

        return total_loss, self.last_metrics

    def anomaly_score(self, z_time, z_freq, mem_R):
        """统一的异常评分，直接替换原来的两个分开的评分"""
        B, T, D = z_time.shape

        P = F.normalize(mem_R.T, p=2, dim=-1).to(z_time.device)

        zt = F.normalize(z_time.reshape(-1, D), p=2, dim=-1)
        zf = F.normalize(z_freq.reshape(-1, D), p=2, dim=-1)

        dist_t = 2 - 2 * zt @ P.T
        dist_f = 2 - 2 * zf @ P.T

        min_dist_t, _ = dist_t.min(dim=-1)
        min_dist_f, _ = dist_f.min(dim=-1)

        assign_t = F.softmax(-dist_t / self.temperature, dim=-1)
        assign_f = F.softmax(-dist_f / self.temperature, dim=-1)
        assign_conflict = 1 - torch.sum(assign_t * assign_f, dim=-1)

        score = (min_dist_t + min_dist_f + 0.3 * assign_conflict).reshape(B, T)

        return score


