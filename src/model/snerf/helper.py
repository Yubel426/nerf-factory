# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn.functional as F


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :]


def sample_along_rays(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    coords = cast_rays(t_vals, rays_o, rays_d)

    return t_vals, coords


def pos_enc(x, min_deg, max_deg):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):

    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, acc, weights, depth


def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        dim=-1,
    )

    s = 1 / num_samples
    if randomized:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def sample_pdf(bins, weights, origins, directions, t_vals, num_samples, randomized):

    t_samples = sorted_piecewise_constant_pdf(
        bins, weights, num_samples, randomized
    ).detach()
    t_vals = torch.sort(torch.cat([t_vals, t_samples], dim=-1), dim=-1).values
    coords = cast_rays(t_vals, origins, directions)
    return t_vals, coords


def normalize(x, eps=1e-5):
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def l2_normalize(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""

    return x / torch.sqrt(
        torch.fmax(torch.sum(x**2, dim=-1, keepdims=True), torch.full_like(x, eps))
    )


def get_rays_uvst(rays_o, rays_d, z1=0, z2=1):
    # rays_o[batch_size, 3], rays_d[batch_size, 3]
    x0, y0, z0 = torch.split(rays_o, 1, dim=-1)
    a, b, c = torch.split(rays_d, 1, dim=-1)
    # 计算交点
    t1 = (z1 - z0) / c
    t2 = (z2 - z0) / c
    x1 = x0 + a * t1
    x2 = x0 + a * t2
    y1 = y0 + b * t1
    y2 = y0 + b * t2
    uvst = torch.cat([x1, y1, x2, y2], dim=-1)
    return uvst

    
def get_interest_point(uvst, z_val):
    # uvst[batch_size, 4]
    # z_val[batch_size, 1]
    x1, y1, x2, y2 = torch.split(uvst, 1, dim=-1)
    # 计算交点
    # z1 = 0, z2 = 1
    x0 = (x2 - x1) * z_val + x1
    y0 = (y2 - y1) * z_val + y1
    interest_point = torch.cat([x0, y0, z_val], dim=-1)
    return interest_point


def get_rays_d(uvst, z1=0, z2=1):
    # uvst[batch_size, 4]
    x1, y1, x2, y2 = torch.split(uvst, 1, dim=-1)
    # 计算光线方向
    a = x2 - x1
    b = y2 - y1
    c = z2 - z1
    c = torch.ones_like(a) * c
    # 方向归一化
    rays_d = normalize(torch.cat([a, b, c], dim=-1))
    return rays_d


def normal_loss(normal):
    normal = torch.sum(normal, dim=0)
    normal = torch.abs(normal)
    max = torch.max(normal)
    loss = torch.sum(normal) - max
    return loss


def refractive(viewdirs, normals, index):
    n_dot_w = torch.sum(
        normals * viewdirs, axis=-1, keepdims=True)
    viewout = (- index * n_dot_w - torch.sqrt(1 - index ** 2 * (1 - n_dot_w ** 2))) * normals + index * viewdirs
    return l2_normalize(viewout)

if __name__ == "__main__":
    v_in = torch.tensor([0, torch.sqrt(torch.tensor(3))/2, 1/2])
    normal = torch.tensor([0,-1,0])
    v_out = refractive(-v_in, -normal,1)
    print(v_out)
