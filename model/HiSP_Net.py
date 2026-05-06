import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_mincut_pool
from .ResMLP import ResMLP,ResMLP_2
from .Cal_Topo import cal_topo



class SpectralClustering(nn.Module):
    def __init__(self, args,features_num, node_num, n_clusters,
                 activation, dropout,):
        super().__init__()

        self.num_heads = args.sc_heads
        self.dropout = dropout

        self.sc_layers = sc_layer_mh(args, features_num, node_num, n_clusters, args.sc_heads, activation, self.dropout)

        self.dim_reduction = nn.Sequential(
            nn.Linear(features_num, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 8),
            nn.LeakyReLU(),
        )

        total_cluster_layers = 0
        for _ in n_clusters:
            total_cluster_layers +=1
        self.fc = nn.Sequential(
            nn.Linear(8 * total_cluster_layers, 2),
        )


        self.ln = nn.LayerNorm(features_num)


        encoder_hidden_size = 32
        self.encoder = nn.Sequential(
            nn.Linear(node_num*node_num, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, node_num*node_num)
        )


    def forward(self,
                corr: torch.tensor,
                node_features: torch.tensor,
                timeseries: torch.tensor,):

        B, N, T = timeseries.shape
        adj = cal_topo(timeseries)
        node_features = self.encoder(node_features.reshape(B,-1)).reshape(B,N,N)

        x_pool_total, mc_loss, o_loss = self.sc_layers(adj, node_features)

        h = x_pool_total

        graph_level_topo = self.dim_reduction(h)
        graph_level_topo = graph_level_topo.reshape(B, -1)
        result = self.fc(graph_level_topo)

        return result, mc_loss, o_loss


def aug_adj_edge_drop_asym(
    adj: torch.Tensor,
    drop_p: float = 0.1,
    only_existing: bool = True,
) -> torch.Tensor:
    if drop_p <= 0.0:
        return adj

    B, N, N2 = adj.shape
    assert N == N2, f"adj must be square, got {adj.shape}"

    adj_aug = adj.clone()

    keep = (torch.rand_like(adj_aug) > drop_p)

    if only_existing:
        keep = keep | (adj_aug <= 0)

    adj_aug = adj_aug * keep.to(adj_aug.dtype)
    return adj_aug

def symmetrize_adj(adj: torch.Tensor, keep_diag: bool = True) -> torch.Tensor:
    adj_sym = 0.5 * (adj + adj.transpose(-1, -2))
    if keep_diag:
        B, N, _ = adj.shape
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        diag = torch.diagonal(adj, dim1=-2, dim2=-1)  # [B,N]
        adj_sym = adj_sym * (1 - eye) + diag.unsqueeze(-1) * eye
    return adj_sym



class sc_layer_mh(nn.Module):
    def __init__(self, args, features_num, node_num, n_clusters, num_heads,
                 activation, dropout):
        super().__init__()
        self.n_heads = num_heads
        self.dropout = dropout
        self.head_fuse = getattr(args, "head_fuse", "gated")  # "mean" or "gated"

        act_dict = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(inplace=True),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
        }
        self.mlp_act = act_dict[activation]

        self.layers = nn.ModuleList()
        self.dim_align = nn.ModuleList()

        self.head_scale = nn.ParameterList()  # (K, C)
        self.head_bias  = nn.ParameterList()  # (K, C)

        self.head_gate = nn.ModuleList()

        for cluster_num in n_clusters:
            block = nn.Sequential(
                ResMLP_2(features_num, cluster_num,
                         nlayers=args.s_mlp_layer,
                         activation=self.mlp_act,
                         dropout=dropout),
                self.mlp_act,
            )
            self.layers.append(block)

            K = self.n_heads
            C = cluster_num
            self.head_scale.append(nn.Parameter(torch.ones(K, C)))   # init=1
            self.head_bias.append(nn.Parameter(torch.zeros(K, C)))   # init=0

            if self.head_fuse == "gated" and K > 1:
                self.head_gate.append(nn.Sequential(
                    nn.Linear(features_num, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, K),
                ))
            else:
                self.head_gate.append(nn.Identity())

            align_block = nn.Sequential(
                nn.Linear(cluster_num, 16),
                nn.LayerNorm(16),
                nn.LeakyReLU(),
                nn.Linear(16, 1),
            )
            self.dim_align.append(align_block)

    def forward(self, adj: torch.Tensor, node_features: torch.Tensor):
        x_pool = node_features         # [B, N, F]
        adj_pool = adj                 # [B, N, N]
        x_pool_total = []
        mc_loss = 0.0
        o_loss = 0.0

        for li, block in enumerate(self.layers):
            mlp, act = block[0], block[1]

            base_logits = act(mlp(x_pool))  # [B, N_cur, C]

            if self.n_heads == 1:
                fused_logits = base_logits
                adj_fused = adj_pool
            else:
                # --- head logits ---
                scale = self.head_scale[li].view(1, self.n_heads, 1, -1)  # [1,K,1,C]
                bias = self.head_bias[li].view(1, self.n_heads, 1, -1)
                head_logits = base_logits.unsqueeze(1) * scale + bias  # [B,K,N_cur,C]

                # --- head weights w: [B,K] ---
                if self.head_fuse == "mean":
                    w = torch.full((x_pool.size(0), self.n_heads), 1.0 / self.n_heads,
                                   device=x_pool.device, dtype=x_pool.dtype)
                elif self.head_fuse == "gated":
                    ctx = x_pool.mean(dim=1)  # [B,F]
                    w = self.head_gate[li](ctx)  # [B,K]
                    w = F.softmax(w, dim=-1)  # [B,K]
                else:
                    raise ValueError(f"Unknown head_fuse: {self.head_fuse}")

                # --- fuse logits ---
                fused_logits = (head_logits * w.view(-1, self.n_heads, 1, 1)).sum(dim=1)  # [B,N_cur,C]

                drop_p = getattr(self, "dropout", 0.1)
                only_existing = getattr(self, "only_existing", True)

                if self.training:
                    adj_heads = []
                    for k in range(self.n_heads):
                        adj_k = aug_adj_edge_drop_asym(adj_pool, drop_p=drop_p, only_existing=only_existing)
                        adj_k = symmetrize_adj(adj_k, keep_diag=True)
                        adj_heads.append(adj_k)

                    adj_heads = torch.stack(adj_heads, dim=1)  # [B,K,N_cur,N_cur]
                    adj_fused = (adj_heads * w.view(-1, self.n_heads, 1, 1)).sum(dim=1)  # [B,N_cur,N_cur]
                else:
                    adj_fused = adj_pool

            x_pool, adj_pool, mc, o = dense_mincut_pool(
                x_pool, adj_fused, fused_logits
            )
            x_pool_total.append(x_pool)
            mc_loss = mc_loss + mc
            o_loss = o_loss + o

        for i, block in enumerate(self.dim_align):
            x_pool_total[i] = block(x_pool_total[i].transpose(1, 2)).squeeze(-1)  # [B, F]
        x_pool_total = torch.stack(x_pool_total, dim=1)  # [B, L, F]

        return x_pool_total, mc_loss, o_loss
