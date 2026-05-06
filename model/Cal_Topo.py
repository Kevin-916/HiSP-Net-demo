import torch
import torch.nn.functional as F


def cal_topo(timeseries):
    B, N, T = timeseries.shape
    mean = timeseries.mean(dim=-1, keepdim=True)
    std = timeseries.std(dim=-1, keepdim=True)
    embeddings = (timeseries - mean) / (std + 1e-8)
    cosine = embeddings @ embeddings.transpose(-1, -2) / (T-1)
    cosine = top_k_dense(cosine, 80)
    adj = normalize_adj_dense(cosine)
    adj = torch.abs(adj)

    return adj



def normalize_adj_dense(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    A = F.relu(A)
    B, N, _ = A.shape
    I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
    A = A + I
    deg = A.sum(dim=-1).clamp(min=eps)
    inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag_embed(inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def top_k_dense(S: torch.Tensor, k: int, include_self: bool = False, fill_val: float = 1e-6) -> torch.Tensor:
    N = S.size(-1)
    kk = k + 1 if include_self else k

    if not include_self:
        if S.dim() == 2:
            mask = torch.eye(N, device=S.device, dtype=torch.bool)
        else:  # [B, N, N]
            mask = torch.eye(N, device=S.device, dtype=torch.bool).unsqueeze(0)
        S = S.masked_fill(mask, 1e-6)

    vals, idx = torch.topk(S, k=kk, dim=-1)

    out = torch.full_like(S, fill_val)
    out.scatter_(dim=-1, index=idx, src=vals)
    return out
