"""
Contrastive Learning components for dimension-aware self-supervised learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    Maps features to a lower-dimensional space for contrastive loss
    """
    def __init__(self, input_dim, output_dim=32, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: (B, T, D) or (B, D)
        returns: (B, T, output_dim) or (B, output_dim)
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            x = self.net(x)
            x = x.reshape(B, T, -1)
        else:
            x = self.net(x)
        return x


def time_mask(x, mask_ratio=0.15):
    """
    Randomly mask time steps (random points)
    x: (B, T, C)
    """
    B, T, C = x.shape
    num_mask = int(T * mask_ratio)

    x_masked = x.clone()
    for b in range(B):
        mask_indices = torch.randperm(T, device=x.device)[:num_mask]
        x_masked[b, mask_indices] = 0.0

    return x_masked


def segment_mask(x, mask_ratio=0.15, min_seg_len=5):
    """
    Segment Mask: 掩盖连续的时间段，所有变量同时掩
    适合 BiMamba 时域分支
    
    x: (B, T, C)
    
    举例（mask_ratio=0.15, T=100）：
      掩盖长度 = 15
      随机选起始点，连续 15 个时间步全部置0
      所有变量同时掩（因为BiMamba学的是时间关系）
    """
    B, T, C = x.shape
    x_masked = x.clone()

    seg_len = max(min_seg_len, int(T * mask_ratio))

    for i in range(B):
        max_start = T - seg_len
        if max_start <= 0:
            start = 0
        else:
            start = torch.randint(0, max_start, (1,)).item()

        # 连续片段置0，所有变量同时掩
        x_masked[i, start:start + seg_len, :] = 0

    return x_masked


def channel_mask(x, mask_ratio=0.15):
    """
    Randomly mask channels
    x: (B, T, C)
    """
    B, T, C = x.shape
    num_mask = max(1, int(C * mask_ratio))

    x_masked = x.clone()
    for b in range(B):
        mask_indices = torch.randperm(C, device=x.device)[:num_mask]
        x_masked[b, :, mask_indices] = 0.0

    return x_masked


def infonce_loss(z1, z2, temperature=0.1):
    """
    InfoNCE contrastive loss
    z1, z2: (B, T, D) positive pairs
    """
    B, T, D = z1.shape

    # Flatten to (B*T, D)
    z1 = z1.reshape(-1, D)
    z2 = z2.reshape(-1, D)

    # Normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Compute similarity matrix
    N = z1.shape[0]

    # For efficiency, use a simplified version
    # Positive pairs: z1[i] and z2[i]
    pos_sim = (z1 * z2).sum(dim=-1) / temperature  # (N,)

    # Negative pairs: z1[i] and z2[j] for j != i
    # Use z2 as negatives
    neg_sim = torch.mm(z1, z2.t()) / temperature  # (N, N)

    # Create labels (diagonal is positive)
    labels = torch.arange(N, device=z1.device)

    # Cross entropy loss
    loss = F.cross_entropy(neg_sim, labels)

    return loss


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent loss (Normalized Temperature-scaled Cross Entropy)
    Used in SimCLR
    z1, z2: (B, D) or (B, T, D)
    """
    if z1.dim() == 3:
        B, T, D = z1.shape
        z1 = z1.reshape(-1, D)
        z2 = z2.reshape(-1, D)

    # Normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    N = z1.shape[0]

    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)

    # Mask out self-similarity
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Labels: positive pairs are at positions N apart
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(z.device)

    loss = F.cross_entropy(sim, labels)

    return loss


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal contrastive loss that encourages similar representations
    for nearby time steps and different representations for distant time steps
    """
    def __init__(self, temperature=0.1, window_size=5):
        super().__init__()
        self.temperature = temperature
        self.window_size = window_size

    def forward(self, z):
        """
        z: (B, T, D) temporal features
        """
        B, T, D = z.shape
        z = F.normalize(z, dim=-1)

        # Create positive pairs from nearby time steps
        loss = 0.0
        count = 0

        for offset in range(1, min(self.window_size + 1, T)):
            z1 = z[:, :-offset]  # (B, T-offset, D)
            z2 = z[:, offset:]   # (B, T-offset, D)

            # Positive similarity
            pos_sim = (z1 * z2).sum(dim=-1) / self.temperature  # (B, T-offset)

            # All pairs in batch as negatives
            z1_flat = z1.reshape(-1, D)  # (B*(T-offset), D)
            z2_flat = z2.reshape(-1, D)

            neg_sim = torch.mm(z1_flat, z2_flat.t()) / self.temperature

            # Labels
            labels = torch.arange(z1_flat.shape[0], device=z.device)

            loss += F.cross_entropy(neg_sim, labels)
            count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=z.device)


class ChannelContrastiveLoss(nn.Module):
    """
    Channel-wise contrastive loss
    Encourages learning channel-specific patterns
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (B, T, C, D) channel-wise features
        """
        B, T, C, D = z1.shape

        # Aggregate over time
        z1 = z1.mean(dim=1)  # (B, C, D)
        z2 = z2.mean(dim=1)

        # Reshape for contrastive loss
        z1 = z1.reshape(B * C, D)
        z2 = z2.reshape(B * C, D)

        return infonce_loss(
            z1.unsqueeze(1),
            z2.unsqueeze(1),
            self.temperature
        )
