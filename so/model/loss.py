#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import dgl


def kl_div(mu, var):
    return kl_divergence(Normal(mu, var.sqrt()),
                         Normal(torch.zeros_like(mu), torch.ones_like(var))).sum(dim=1).mean()


def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def _median_heuristic_sigma(z, subsample=1000):
    """Estimate RBF sigma using median heuristic.
    Uses torch.pdist to get condensed pairwise distances (upper triangle),
    which is memory-efficient compared to forming full (N,N) distance matrix.
    Returns scalar float sigma (bandwidth).
    """
    N = z.size(0)
    if N == 0:
        return 1.0
    if N > subsample:
        idx = torch.randperm(N, device=z.device)[:subsample]
        z_s = z[idx]
    else:
        z_s = z

    # pdist returns pairwise distances (condensed upper triangle)
    if z_s.size(0) < 2:
        # not enough points to compute distances
        return 1.0

    d = torch.pdist(z_s, p=2)  # distances, not squared
    if d.numel() == 0:
        return 1.0

    median_dist = float(d.median().item())
    # avoid degenerate tiny sigma
    if median_dist < 1e-12:
        return 1.0
    return median_dist


def rbf_kernel(x, y=None, sigma=None):
    """Compute RBF kernel matrix between x and y: K_ij = exp(-||x_i-y_j||^2 / (2*sigma^2))
    - If y is None, computes between x and itself.
    - sigma can be python float or a tensor; it will be converted to tensor matching x's dtype/device.
    """
    if y is None:
        y = x
    # ensure contiguous for efficient matmul
    x = x.contiguous()
    y = y.contiguous()

    if sigma is None:
        # estimate from concatenation (subsample inside)
        xy = torch.cat([x, y], dim=0)
        sigma = _median_heuristic_sigma(xy)

    # convert sigma to tensor on same device/dtype for safe math
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    else:
        sigma = sigma.to(device=x.device, dtype=x.dtype)

    # squared distances via efficient matmul formula
    x2 = (x ** 2).sum(dim=1, keepdim=True)   # (n_x, 1)
    y2 = (y ** 2).sum(dim=1, keepdim=True)   # (n_y, 1)
    dist2 = x2 + y2.t() - 2.0 * (x @ y.t())  # (n_x, n_y)
    dist2 = torch.clamp(dist2, min=0.0)

    denom = 2.0 * (sigma ** 2) + 1e-8  # small eps for safety
    K = torch.exp(-dist2 / denom)
    return K


def mmd_between_sets(x, y, sigma=None):
    """Compute biased MMD^2 between x and y using RBF kernel (biased estimator = means).
    Returns a scalar tensor (non-negative, clamped).
    """
    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)
    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return torch.clamp(mmd2, min=0.0)


def mmd_loss_to_reference(z, batch_idx, reference_batch=None, sigma=None,
                          subsample_per_batch=500, mode="to_reference"):
    """
    Compute average MMD^2 between a chosen reference batch and every other batch in the minibatch.
    - z: (N, D) tensor
    - batch_idx: (N,) integer tensor labeling batches
    - reference_batch: None or int (if None, pick first unique)
    - subsample_per_batch: cap per-batch samples to reduce cost
    Returns: scalar tensor average MMD^2
    """
    device = z.device
    dtype = z.dtype

    unique_batches = torch.unique(batch_idx)
    B = unique_batches.numel()
    if B <= 1:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # choose reference
    if reference_batch is None:
        ref = int(unique_batches[0].item())
    else:
        ref = int(reference_batch)

    # prepare per-batch sampling function
    def subsample_tensor(x, cap):
        n = x.size(0)
        if n > cap:
            idx = torch.randperm(n, device=x.device)[:cap]
            return x[idx]
        return x

    # get reference samples (and fallback if empty)
    ref_mask = (batch_idx == ref)
    z_ref_all = z[ref_mask]
    if z_ref_all.size(0) == 0:
        # fallback to first unique if specified reference empty
        ref = int(unique_batches[0].item())
        ref_mask = (batch_idx == ref)
        z_ref_all = z[ref_mask]

    z_ref = subsample_tensor(z_ref_all, subsample_per_batch)

    # compute sigma from subsampled points if not provided (cheaper than full)
    if sigma is None:
        # combine small subsample of each batch for robust estimate
        # here we sample up to 2*subsample_per_batch points total (ref + a random other set)
        sample_pool = [z_ref]
        # try adding one other batch (if available) to avoid single-batch estimation bias
        for bi in unique_batches:
            bi = int(bi.item())
            if bi == ref:
                continue
            xi = z[batch_idx == bi]
            if xi.numel() > 0:
                sample_pool.append(subsample_tensor(xi, subsample_per_batch))
                break
        sigma = _median_heuristic_sigma(torch.cat(sample_pool, dim=0))

    loss_sum = z_ref.new_tensor(0.0)
    count = 0

    # iterate other batches
    for bi in unique_batches:
        bi = int(bi.item())
        if bi == ref:
            continue
        xi = z[batch_idx == bi]
        if xi.size(0) == 0:
            continue
        xi = subsample_tensor(xi, subsample_per_batch)
        mmd2 = mmd_between_sets(z_ref, xi, sigma=sigma)
        loss_sum = loss_sum + mmd2  # keep as tensor for grad
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return (loss_sum / float(count)).to(device=device, dtype=dtype)


def graph_rl(block, z, device):
    # 1. positive
    # ======================
    pos_src, pos_dst = block.edges()
    pos_src = pos_src.to(device)
    pos_dst = pos_dst.to(device)
    # remove self-loop
    mask = pos_src != pos_dst
    pos_src = pos_src[mask]
    pos_dst = pos_dst[mask]
    if pos_src.numel() == 0:
        struct_loss = torch.tensor(0.0, device=device)
        return struct_loss
    else:
        k = len(pos_src)
        neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(block, k)
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)
        # 3. loss
        pos_logits = torch.sum(z[pos_src] * z[pos_dst], dim=1)
        neg_logits = torch.sum(z[neg_src] * z[neg_dst], dim=1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        struct_loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return struct_loss


def cov_offdiag_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    z: [B, D]
    return: scalar
    """
    B, D = z.shape
    zc = z - z.mean(dim=0, keepdim=True)                 # center
    cov = (zc.T @ zc) / (B - 1 + eps)                    # [D, D]
    offdiag = cov - torch.diag(torch.diag(cov))
    return (offdiag ** 2).sum() / D                      # /D 让尺度更稳定


def var_floor_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Encourage per-dimension std >= gamma
    """
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)  # [D]
    return torch.relu(gamma - std).mean()


def barlow_twins_loss_single_view(z: torch.Tensor, lambd_offdiag: float = 0.005, eps: float = 1e-4) -> torch.Tensor:
    """
    Single-view variant: encourage correlation matrix ~ I
    """
    B, D = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)
    zn = zc / std                                        # z-score

    corr = (zn.T @ zn) / (B - 1 + eps)                   # [D, D], approx correlation
    on_diag = (torch.diag(corr) - 1).pow(2).sum()
    off_diag = (corr - torch.diag(torch.diag(corr))).pow(2).sum()
    return (on_diag + lambd_offdiag * off_diag) / D


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
