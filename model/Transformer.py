from math import sqrt



import torch



import torch.nn as nn



import torch.nn.functional as F



from einops import rearrange



import numpy as np







from model.embedding import InputEmbedding



from model.bimamba_encoder import BiMambaTemporalEncoder

from model.contrastive import ProjectionHead, segment_mask, channel_mask, infonce_loss







from utils.dataplot import plot_time_series_comparison



from model.RevIN import RevIN











class Decoder(nn.Module):



    def __init__(self, w_size, d_model, c_out, networks=['linear'], n_layers=1,



                 group_embedding='False', kernel_size=[1], patch_size=-1, activation='gelu', dropout=0.0, device='cpu'): 



        super(Decoder, self).__init__()







        self.decoder = InputEmbedding(in_dim=d_model, d_model=c_out, n_window=w_size,



                                      dropout=dropout, n_layers=n_layers,



                                      branch_layers=networks,



                                      match_dimension='last',



                                      group_embedding=group_embedding,



                                      kernel_size=kernel_size, init_type='normal',



                                      device=device)







    def forward(self, x):



        """



        x : N x L x C(=d_model)



        """



        out = self.decoder(x)



        return out  # N x L x c_out











class TransformerVar(nn.Module):







    DEFAULTS = {}







    def __init__(self, config, n_heads=1, d_ff=128, dropout=0.3, activation='gelu', gain=0.02):



        super(TransformerVar, self).__init__()







        self.__dict__.update(TransformerVar.DEFAULTS, **config)







        # Encoding



        branch1_group = self.branches_group_embedding.split('_')[0]



        branch2_group = self.branches_group_embedding.split('_')[1]







        branch1_dim = self.input_c if self.branch1_match_dimension == 'none' else self.d_model



        branch2_dim = self.input_c if self.branch2_match_dimension == 'none' else self.d_model







        # ====== t-Encoder: switch between original and BiMamba-enhanced ======



        use_our_embedding = getattr(self, 'use_our_embedding', 'False') == 'True'



        pure_mamba = getattr(self, 'pure_mamba', 'False') == 'True'



        if use_our_embedding:



            self.encoder_branch1 = BiMambaTemporalEncoder(in_dim=self.input_c, d_model=branch1_dim, n_window=self.win_size,



                                                dropout=dropout, n_layers=self.num_layers,



                                                branch_layers=self.branch1_networks,



                                                match_dimension=self.branch1_match_dimension,



                                                group_embedding=branch1_group,



                                                kernel_size=self.multiscale_kernel_size, init_type=self.embedding_init,



                                                device=self.device,



                                                pure_mamba=pure_mamba)



        else:



            self.encoder_branch1 = InputEmbedding(in_dim=self.input_c, d_model=branch1_dim, n_window=self.win_size,



                                                  dropout=dropout, n_layers=self.encoder_layers,



                                                  branch_layers=self.branch1_networks,



                                                  match_dimension=self.branch1_match_dimension,



                                                  group_embedding=branch1_group,



                                                  kernel_size=self.multiscale_kernel_size, init_type=self.embedding_init,



                                                  device=self.device)







        # ====== i-Encoder: switch between original and Wavelet+Transformer ======



        use_wavelet_branch2 = getattr(self, 'use_wavelet_branch2', 'False') == 'True'



        if use_wavelet_branch2:



            self.encoder_branch2 = OurEmbedding_Branch2(



                in_dim=self.input_c, d_model=branch2_dim, n_window=self.win_size,



                dropout=dropout, n_layers=self.encoder_layers,



                branch_layers=self.branch2_networks,



                match_dimension=self.branch2_match_dimension,



                group_embedding=branch2_group,



                kernel_size=self.multiscale_kernel_size,



                init_type=self.embedding_init, device=self.device,



                wavelet_level=getattr(self, 'wavelet_level', 3),



                n_heads=getattr(self, 'n_heads', 4),



                axial_layers=getattr(self, 'axial_layers', 1),



                attn_type=getattr(self, 'attn_type', 'scale')



            )



        else:



            self.encoder_branch2 = InputEmbedding(in_dim=self.input_c, d_model=branch2_dim, n_window=self.win_size,



                                                  dropout=dropout, n_layers=self.encoder_layers,



                                                  branch_layers=self.branch2_networks,



                                                  match_dimension=self.branch2_match_dimension,



                                                  group_embedding=branch2_group,



                                                  kernel_size=self.multiscale_kernel_size,



                                                  init_type=self.embedding_init, device=self.device)







        self.activate_func = nn.GELU()







        self.dropout = nn.AlphaDropout(p=dropout)







        self.loss_func = nn.MSELoss(reduction='none')







        # 原型记忆库 (Prototype Memory)



        self.use_memory = getattr(self, 'use_memory', 'True') == 'True'



        if self.use_memory:



            self.mem_R, self.mem_I = create_memory_matrix(N=branch2_dim,



                                                          L=self.win_size,



                                                          mem_type=self.memory_guided,



                                                          option='options2')



        else:



            self.mem_R = None



            self.mem_I = None







        branch1_out_dim = self.output_c if self.branch1_match_dimension == 'none' else self.d_model







        # After feature_prj, dimension becomes output_c, so decoder input should be output_c



        decoder_in_dim = self.output_c







        # ====== 时域解码器 (t-Decoder) ======



        self.weak_decoder = Decoder(w_size=self.win_size,



                                    d_model=decoder_in_dim,



                                    c_out=self.output_c,



                                    networks=self.decoder_networks,



                                    n_layers=self.decoder_layers,



                                    group_embedding=self.decoder_group_embedding,



                                    kernel_size=self.multiscale_kernel_size,



                                    activation='gelu',



                                    dropout=0.0,       # The dropout in decoder is set as zero



                                    device=self.device)







        if self.branch1_match_dimension == 'none':



            self.feature_prj = lambda x: x



        else:



            self.feature_prj = nn.Linear(branch1_out_dim, self.output_c)



        



        # ====== 双分支对称重构 (Dual Branch Symmetric Reconstruction) ======



        self.use_dual_recon = getattr(self, 'use_dual_recon', 'False') == 'True'



        if self.use_dual_recon:



            branch2_out_dim = self.output_c if self.branch2_match_dimension == 'none' else self.d_model



            



            # 频域解码器 (f-Decoder): 从 i_query 重构



            self.freq_decoder = Decoder(w_size=self.win_size,



                                        d_model=decoder_in_dim,



                                        c_out=self.output_c,



                                        networks=self.decoder_networks,



                                        n_layers=self.decoder_layers,



                                        group_embedding=self.decoder_group_embedding,



                                        kernel_size=self.multiscale_kernel_size,



                                        activation='gelu',



                                        dropout=0.0,



                                        device=self.device)



            



            # 频域特征投影



            if self.branch2_match_dimension == 'none':



                self.freq_feature_prj = lambda x: x



            else:



                self.freq_feature_prj = nn.Linear(branch2_out_dim, self.output_c)



            



            # 融合权重 (可学习)



            self.recon_weight_t = nn.Parameter(torch.tensor(0.5))



            self.recon_weight_f = nn.Parameter(torch.tensor(0.5))







        # ====== Contrastive Learning Module ======



        self.use_contrastive = getattr(self, 'use_contrastive', 'False') == 'True'



        DATASET_ADAPTIVE_CL_WEIGHTS = {
            'SMAP': (0.01, 0.01),
            'MSL': (0.03, 0.02),
            'PSM': (0.05, 0.05),
            'SMD': (0.04, 0.04),
            'SWaT': (0.03, 0.03),
            'NIPS_TS_Water': (0.08, 0.08),
            'NIPS_TS_Swan': (0.07, 0.07)
        }

        def resolve_dataset_cl_weights(dataset_name, lambda_cl_time, lambda_cl_freq, auto_cl_dataset):

            if not auto_cl_dataset:

                return lambda_cl_time, lambda_cl_freq

            if dataset_name in DATASET_ADAPTIVE_CL_WEIGHTS:

                return DATASET_ADAPTIVE_CL_WEIGHTS[dataset_name]

            return lambda_cl_time, lambda_cl_freq

        if self.use_contrastive:

            proj_dim = getattr(self, 'proj_dim', 32)

            # t-Encoder projection head (temporal contrastive)

            self.temporal_proj = ProjectionHead(branch1_dim, proj_dim)

            # i-Encoder projection head (frequency contrastive)

            self.wavelet_proj = ProjectionHead(branch2_dim, proj_dim)

            lambda_cl_time = getattr(self, 'lambda_cl_time', 0.1)

            lambda_cl_freq = getattr(self, 'lambda_cl_freq', 0.1)

            auto_cl_dataset = getattr(self, 'auto_cl_dataset', 'True') == 'True'

            dataset_name = getattr(self, 'dataset', None)

            self.lambda_cl_time, self.lambda_cl_freq = resolve_dataset_cl_weights(
                dataset_name, lambda_cl_time, lambda_cl_freq, auto_cl_dataset
            )

            self.base_lambda_cl_time = self.lambda_cl_time

            self.base_lambda_cl_freq = self.lambda_cl_freq

            self.runtime_lambda_cl_time = self.lambda_cl_time

            self.runtime_lambda_cl_freq = self.lambda_cl_freq

            self.mask_ratio = getattr(self, 'mask_ratio', 0.15)

            self.set_contrastive_runtime_state(enabled=True)

        # ====== ProtoSVDD Module ======

        # 不对称ProtoSVDD：频域轻量对齐，时域对齐到原型

        self.use_svdd = getattr(self, 'use_svdd', 'False') == 'True'

        self.use_proto_svdd = self.use_svdd  # 统一使用新架构

        if self.use_svdd:

            svdd_proj_dim = getattr(self, 'svdd_proj_dim', 64)

            # 频域投影头：非常轻量，几乎不改变原始频域特征

            self.svdd_proj_freq = nn.Sequential(
                nn.LayerNorm(branch2_dim),
                nn.Linear(branch2_dim, svdd_proj_dim)
            )

            # 时域投影头：唯一需要训练的部分，对齐到频域原型

            self.svdd_proj_time = nn.Sequential(
                nn.Linear(branch1_dim, branch1_dim),
                nn.GELU(),
                nn.LayerNorm(branch1_dim),
                nn.Linear(branch1_dim, svdd_proj_dim)
            )



    def set_contrastive_runtime_state(self, enabled=True):

        if not self.use_contrastive:
            return

        if enabled:
            self.runtime_lambda_cl_time = self.base_lambda_cl_time
            self.runtime_lambda_cl_freq = self.base_lambda_cl_freq
        else:
            self.runtime_lambda_cl_time = 0.0
            self.runtime_lambda_cl_freq = 0.0



    def forward(self, input_data, mode='train'):

        """



        x (input time window) : B x L x enc_in



        """

        z1 = z2 = input_data



        t_query, t_latent_list = self.encoder_branch1(z1)



        i_query, _ = self.encoder_branch2(z2)



        # 原型记忆库注意力计算 (Prototype Memory Attention)



        if self.use_memory:

            # use dot production with static sinusoid basis

            mem = self.mem_R.T.to(self.device)

            # differencing_q = (i_query - torch.roll(i_query, shifts=1, dims=-2))

            # It seems that using differencing is better than using i_query

            attn = torch.einsum('blf,jl->bfj', i_query, self.mem_R.to(self.device).detach())

            attn = torch.softmax(attn / self.temperature, dim=-1)



        else:

            mem = None

            attn = None



        queries = i_query



        # ====== 时域重构 (Temporal Reconstruction) ======

        combined_z_t = self.feature_prj(t_query)



        out_t, _ = self.weak_decoder(combined_z_t)



        # ====== 双分支对称重构 (Dual Branch Symmetric Reconstruction) ======

        if self.use_dual_recon:

            # 频域重构 (Frequency Reconstruction)

            combined_z_f = self.freq_feature_prj(i_query)



            out_f, _ = self.freq_decoder(combined_z_f)



            # 自适应融合权重 (归一化)

            w_t = torch.sigmoid(self.recon_weight_t)

            w_f = torch.sigmoid(self.recon_weight_f)

            w_sum = w_t + w_f + 1e-8

            w_t = w_t / w_sum

            w_f = w_f / w_sum



            # 融合重构输出

            out = w_t * out_t + w_f * out_f

        else:

            out = out_t

            out_f = None



        # ====== Contrastive Learning (training only) ======

        cl_loss = torch.tensor(0.0, device=input_data.device)

        compute_cl = False

        if self.use_contrastive:

            if mode == 'train':

                compute_cl = (self.runtime_lambda_cl_time > 0) or (self.runtime_lambda_cl_freq > 0)

            elif mode == 'vali_cl':

                compute_cl = (self.base_lambda_cl_time > 0) or (self.base_lambda_cl_freq > 0)

        if compute_cl:

            cl_loss = self._compute_contrastive_loss(input_data, t_query, i_query)



        # ====== CD-SVDD Projections ======

        z_svdd_time = None

        z_svdd_freq = None

        if self.use_svdd:

            # 投影到 SVDD 空间

            B, T, D = t_query.shape

            z_svdd_time = self.svdd_proj_time(t_query.reshape(-1, D)).reshape(B, T, -1)

            z_svdd_freq = self.svdd_proj_freq(i_query.reshape(-1, i_query.size(-1))).reshape(B, T, -1)



        return {"out": out, "out_t": out_t, "out_f": out_f, "queries": queries, "mem": mem, "attn": attn,

                "cl_loss": cl_loss, "z_svdd_time": z_svdd_time, "z_svdd_freq": z_svdd_freq}







    def _compute_contrastive_loss(self, x, t_query, i_query):

        """



        Compute contrastive loss for both branches



        



        BiMamba (t-Encoder): Segment Mask - learn temporal dependencies



        Wavelet (i-Encoder): Channel Mask - learn inter-variable correlations



        """



        cl_loss = torch.tensor(0.0, device=x.device)



        



        # ====== Temporal Contrastive (BiMamba branch) ======



        lambda_cl_time = self.runtime_lambda_cl_time if self.training else self.base_lambda_cl_time



        lambda_cl_freq = self.runtime_lambda_cl_freq if self.training else self.base_lambda_cl_freq



        if lambda_cl_time > 0:



            x_seg = segment_mask(x, self.mask_ratio)



            t_query_aug, _ = self.encoder_branch1(x_seg)



            



            # Pool over time: (B, T, D) -> (B, D)



            z_time = t_query.mean(dim=1)



            z_time_aug = t_query_aug.mean(dim=1)



            



            # Project



            z_time = self.temporal_proj(z_time)



            z_time_aug = self.temporal_proj(z_time_aug)



            



            # InfoNCE loss



            L_cl_time = infonce_loss(z_time.unsqueeze(1), z_time_aug.unsqueeze(1))



            cl_loss = cl_loss + lambda_cl_time * L_cl_time



        



        # ====== Frequency Contrastive (Wavelet branch) ======



        if lambda_cl_freq > 0:



            x_chan = channel_mask(x, self.mask_ratio)



            i_query_aug, _ = self.encoder_branch2(x_chan)



            



            # Pool over time



            z_freq = i_query.mean(dim=1)



            z_freq_aug = i_query_aug.mean(dim=1)



            



            # Project



            z_freq = self.wavelet_proj(z_freq)



            z_freq_aug = self.wavelet_proj(z_freq_aug)



            



            # InfoNCE loss



            L_cl_freq = infonce_loss(z_freq.unsqueeze(1), z_freq_aug.unsqueeze(1))



            cl_loss = cl_loss + lambda_cl_freq * L_cl_freq



        



        return cl_loss







    def get_attn_score(self, query, key, scale=None):



        """



        Calculating attention score with sparsity regularization



        query (initial features) : (NxL) x C or N x C -> T x C



        key (memory items): M x C



        """



        scale = 1. / sqrt(query.size(-1)) if scale is None else 1. / scale







        attn = torch.matmul(query, torch.t(key.to(self.device)))  # (TxC) x (CxM) -> TxM







        attn = attn * scale







        # attn = F.softmax(attn / self.temperature, dim=-1)



        # attn = torch.einsum('tl,kfl->tkf', query, key.to(self.device))  # (TxC) x (CxM) -> TxM



        # attn = attn.max(dim=1)[0]







        return attn







def generate_rolling_matrix(input_matrix):



    F, L = input_matrix.size()



    # Initialize an empty tensor of shape [L, F, L] to store the result



    output_matrix = torch.empty(L, F, L)







    # Iterate over each step from 0 to L-1



    for step in range(L):



        # Roll the rows of the input tensor along the last dimension



        rolled_matrix = input_matrix.roll(shifts=step, dims=1)



        # Assign the rolled tensor to the appropriate slice in the output tensor



        output_matrix[step] = rolled_matrix







    return output_matrix







def create_memory_matrix(N, L, mem_type='sinusoid', option='option1'):







    with torch.no_grad():



        if mem_type  == 'sinusoid' or mem_type  == 'cosine_only':



            row_indices = torch.arange(N).reshape(-1, 1)



            col_indices = torch.arange(L)



            grid = row_indices * col_indices



            # Calculate the period values using the grid



            init_matrix_r = torch.cos((1 / L) * 2 * torch.tensor([torch.pi]) * grid)



            init_matrix_i = torch.sin((1 / L) * 2 * torch.tensor([torch.pi]) * grid)



        elif mem_type  == 'uniform' or mem_type  == 'uniform_only':



            init_matrix_r = torch.rand((N, L), dtype=torch.float)



            init_matrix_i = torch.rand((N, L), dtype=torch.float)



        elif mem_type  == 'orthogonal_uniform' or mem_type  == 'orthogonal_uniform_only':



            init_matrix_r = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))



            init_matrix_i = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))



        elif mem_type  == 'normal' or mem_type  == 'normal_only':



            init_matrix_r = torch.randn((N, L), dtype=torch.float)



            init_matrix_i = torch.randn((N, L), dtype=torch.float)



        elif mem_type  == 'orthogonal_normal' or mem_type  == 'orthogonal_normal_only':



            init_matrix_r = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))



            init_matrix_i = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))







        # rolling the wave



        if option == 'option4':



            init_matrix_r = generate_rolling_matrix(init_matrix_r)



            init_matrix_i = generate_rolling_matrix(init_matrix_i)







        if 'only' not in mem_type:



            return init_matrix_r, init_matrix_i



        else:



            return init_matrix_r, torch.zeros_like(init_matrix_r)



