import torch
import torch.nn as nn
from model.embedding import TokenEmbedding
from model.attn_layer import PositionalEmbedding
from model.mamba_simple import BiMambaEncoder
from model.Conv_Blocks import Inception_Block


class BiMambaTemporalEncoder(nn.Module):
    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=1,
                 use_pos_embedding='False', group_embedding='False', kernel_size=5,
                 init_type='kaiming', match_dimension='first', branch_layers=['linear'],
                 pure_mamba=False):
        super(BiMambaTemporalEncoder, self).__init__()
        
        self.device = device
        self.d_model = d_model
        self.match_dimension = match_dimension
        self.pure_mamba = pure_mamba
        self.n_layers = n_layers
        
        # Determine output dimension based on match_dimension
        if match_dimension == 'none':
            self.out_dim = in_dim
        else:
            self.out_dim = d_model
        
        if not pure_mamba:
            # ====== Mode 1: ProCSAD + BiMamba fusion ======
            self.token_embedding = TokenEmbedding(
                in_dim=in_dim, d_model=d_model, n_window=n_window,
                n_layers=n_layers, branch_layers=branch_layers,
                group_embedding=group_embedding, match_dimension=match_dimension,
                init_type=init_type, kernel_size=kernel_size,
                dropout=dropout
            )
            
            self.bimamba_branch = BiMambaEncoder(
                input_dim=in_dim,
                d_model=self.out_dim,
                num_layers=self.n_layers,
                d_state=32,
                d_conv=4,
                expand=2,
                dropout=dropout
            )
            
            # Adaptive fusion weight
            self.alpha_raw = nn.Parameter(torch.tensor(-1.0))
        else:
            # ====== Mode 2: Pure BiMamba (no FFT) ======
            # Input projection with multi-scale conv
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size]
            self.input_conv = Inception_Block(
                in_channels=in_dim,
                out_channels=self.out_dim,
                kernel_list=kernel_size,
                groups=1
            )
            
            # BiMamba encoder
            self.bimamba_branch = BiMambaEncoder(
                input_dim=self.out_dim,
                d_model=self.out_dim,
                num_layers=self.n_layers,
                d_state=32,
                d_conv=4,
                expand=2,
                dropout=dropout
            )
        
        self.norm = nn.LayerNorm(self.out_dim)
        self.pos_embedding = PositionalEmbedding(d_model=self.out_dim)
        self.use_pos_embedding = use_pos_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x: (B, L, C_in) input time series
        returns: (output, latent_list)
        """
        x = x.to(self.device)
        latent_list = []
        
        if not self.pure_mamba:
            # ====== Mode 1: ProCSAD + BiMamba fusion ======
            h_time = self.bimamba_branch(x)
            h_freq, latent_list = self.token_embedding(x)
            
            alpha = torch.sigmoid(self.alpha_raw)
            h_out = h_freq + alpha * h_time
        else:
            # ====== Mode 2: Pure BiMamba ======
            # (B, L, C) -> (B, C, L) -> Conv -> (B, out_dim, L) -> (B, L, out_dim)
            h = x.permute(0, 2, 1)
            h = self.input_conv(h)
            h = h.permute(0, 2, 1)
            latent_list.append(h)
            
            h_out = self.bimamba_branch(h)
            latent_list.append(h_out)
        
        h_out = self.norm(h_out)
        
        if self.use_pos_embedding == 'True':
            h_out = h_out + self.pos_embedding(h_out).to(self.device)
        
        return self.dropout(h_out), latent_list
