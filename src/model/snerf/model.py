# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from random import random
from typing import *

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import src.model.snerf.helper as helper
import utils.store_image as store_image
from src.model.interface import LitModel


@gin.configurable()
class ViewMLP(torch.nn.Module):
    def __init__(self):
        super(ViewMLP, self).__init__()
        self.net_activation = nn.Tanh()
        layers = []
        dim_in = [7, 256, 256, 256, 128, 256, 256, 256, 256]
        dim_out = [256, 256, 249, 128, 256, 256, 128, 256, 7]
        self.skip = [2, 6]
        for i in range(9):
            linear = torch.nn.Linear(dim_in[i], dim_out[i])
            init.xavier_uniform_(linear.weight)
            layers.append(linear)
        self.layers = torch.nn.ModuleList(layers)
        del layers

    def forward(self, x):
        inputs = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.net_activation(x)
            if i == 2:
                x = torch.cat([x, inputs], dim=-1)
            if i == 3:
                inputs = x
            if i == 6:
                x = torch.cat([x, inputs], dim=-1)
        # TODO: add the noise
        padding = 0.001
        x = helper.l2_normalize(x)
        x = x * (1 + 2 * padding) - padding
        return x


@gin.configurable()
class SNeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        disable_rgb: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(SNeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        self.density_layer = nn.Linear(netwidth, num_density_channels)
        init.xavier_uniform_(self.density_layer.weight)

        self.grad_layer = nn.Linear(netwidth, 3)
        init.xavier_uniform_(self.grad_layer.weight)

        if not disable_rgb:
            views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
            for idx in range(netdepth_condition - 1):
                layer = nn.Linear(netwidth_condition, netwidth_condition)
                init.xavier_uniform_(layer.weight)
                views_linear.append(layer)

            self.views_linear = nn.ModuleList(views_linear)

            self.bottleneck_layer = nn.Linear(netwidth, netwidth)
            self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

            init.xavier_uniform_(self.bottleneck_layer.weight)
            init.xavier_uniform_(self.rgb_layer.weight)

            self.rgb_activation = nn.Sigmoid()
            self.sigma_activation = nn.ReLU()

    def forward(self, x, condition, comput_normal=False):
        # 1st part
        normals = torch.zeros_like(x)
        x_raw = x
        if comput_normal and self.training:
            with torch.set_grad_enabled(True):
                x.requires_grad = True
                x_to_compute_normal = x
                x = helper.pos_enc(
                x,
                self.min_deg_point,
                self.max_deg_point,
                )
                num_samples, feat_dim = x.shape[1:]
                x = x.reshape(-1, feat_dim)
                inputs = x
                for idx in range(self.netdepth):
                    x = self.pts_linears[idx](x)
                    x = self.net_activation(x)
                    if idx % self.skip_layer == 0 and idx > 0:
                        x = torch.cat([x, inputs], dim=-1)

                raw_density = self.density_layer(x).reshape(
                    -1, num_samples, self.num_density_channels
                )
                normals_pred = self.grad_layer(x).reshape(
                    -1, num_samples, 3
                )
                normals_pred = -helper.l2_normalize(normals_pred)

                raw_density_grad = torch.autograd.grad(
                    outputs=raw_density.sum(), inputs=x_to_compute_normal, retain_graph=True
                )[0]

                raw_density_grad = raw_density_grad.reshape(
                    -1, num_samples, 3
                )

                normals = -helper.l2_normalize(raw_density_grad)
                x_to_compute_normal.detach()
        else:
            x = helper.pos_enc(
                x,
                self.min_deg_point,
                self.max_deg_point,
            )
            num_samples, feat_dim = x.shape[1:]
            x = x.reshape(-1, feat_dim)
            inputs = x
            for idx in range(self.netdepth):
                x = self.pts_linears[idx](x)
                x = self.net_activation(x)
                if idx % self.skip_layer == 0 and idx > 0:
                    x = torch.cat([x, inputs], dim=-1)

            raw_density = self.density_layer(x).reshape(
                -1, num_samples, self.num_density_channels
            )

            normals_pred = self.grad_layer(x).reshape(
                -1, num_samples, 3
            )
            normals_pred = -helper.l2_normalize(normals_pred)

        if self.disable_rgb:
            raw_rgb = torch.zeros_like(x_raw)
            return raw_rgb, raw_density, normals, normals_pred
        
        # 2nd part
        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)
        rgb = self.rgb_activation(raw_rgb)
        density = self.sigma_activation(raw_density)

        return dict(
            rgb = rgb,
            density = density,
            normals = normals,
            normals_pred = normals_pred,
        )


@gin.configurable()
class SNeRF(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,
        noise_std: float = 0.0,
        lindisp: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(SNeRF, self).__init__()

        self.rgb_activation = nn.Sigmoid()
        self.sigma_activation = nn.ReLU()

        self.coarse_mlp1 = SNeRFMLP(min_deg_point, max_deg_point, deg_view)
        self.fine_mlp1 = SNeRFMLP(min_deg_point, max_deg_point, deg_view)

        self.view_mlp = ViewMLP()
    def forward(self, rays, randomized, white_bkgd, near, far):

        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                mlp1 = self.coarse_mlp1
                mlp2 = self.coarse_mlp1
                # 1st part
                t_vals1, samples1 = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=int(self.num_coarse_samples/2),
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                viewdirs_enc1 = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)
                res1 = mlp1( samples1, viewdirs_enc1, True)
                comp_rgb1, acc1, weights1, depth = helper.volumetric_rendering(
                    res1["rgb"],
                    res1["density"],
                    t_vals1,
                    rays["rays_d"],
                    white_bkgd=white_bkgd,
                )
                res1["weights"] = weights1
                if self.training:
                    normal = (weights1[..., None] * res1["normals"]).sum(dim=-2)
                else:
                    normal = (weights1[..., None] * res1["normals_pred"]).sum(dim=-2)

                # 2nd part
                pts1 = rays["rays_o"] + torch.mul(depth.unsqueeze(1), rays["rays_d"])
                pts_z = pts1[..., 2:]
                uvst = helper.get_rays_uvst(rays["rays_o"], rays["rays_d"],0,1)

                uvst_normal = torch.cat([normal, uvst], dim=-1)
                uvst_normal_pred = self.view_mlp(uvst_normal)
                uvst_pred = uvst_normal_pred[..., 3:]
                rays_o_pred = helper.get_interest_point(uvst_pred, pts_z)
                rays_d_pred = helper.get_rays_d(uvst_pred)

                uvst_normal_repred = self.view_mlp(torch.neg(uvst_normal_pred))
                uvst_repred = uvst_normal_repred[..., 3:]

                t_vals2, samples2 = helper.sample_along_rays(
                    rays_o=rays_o_pred,
                    rays_d=rays_d_pred,
                    num_samples=int(self.num_coarse_samples/2),
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                viewdirs_enc2 = helper.pos_enc(rays_d_pred, 0, self.deg_view)
                res2 = mlp2( samples2, viewdirs_enc2)

                comp_rgb2, acc2, weights2, _ = helper.volumetric_rendering(
                    res2["rgb"],
                    res2["density"],
                    t_vals2,
                    rays_d_pred,
                    white_bkgd=white_bkgd,
                )
                res2["weights"] = weights2

            else:
                mlp1 = self.fine_mlp1
                mlp2 = self.fine_mlp1
                # 1st part
                t_mids1 = 0.5 * (t_vals1[..., 1:] + t_vals1[..., :-1])
                t_vals1, samples1 = helper.sample_pdf(
                    bins=t_mids1,
                    weights=weights1[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals1,
                    num_samples=int(self.num_fine_samples/2),
                    randomized=randomized,
                )

                viewdirs_enc1 = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)
                res1 = mlp1( samples1, viewdirs_enc1, True)

                comp_rgb1, acc1, weights1, _ = helper.volumetric_rendering(
                    res1["rgb"],
                    res1["density"],
                    t_vals1,
                    rays["rays_d"],
                    white_bkgd=white_bkgd,
                )
                res1["weights"] = weights1
                if self.training:
                    normal = (weights1[..., None] * res1["normals"]).sum(dim=-2)
                else:
                    normal = (weights1[..., None] * res1["normals_pred"]).sum(dim=-2)

                # 2nd part
                t_mids2 = 0.5 * (t_vals2[..., 1:] + t_vals2[..., :-1])
                t_vals2, samples2 = helper.sample_pdf(
                    bins=t_mids2,
                    weights=weights2[..., 1:-1],
                    origins=rays_o_pred,
                    directions=rays_d_pred,
                    t_vals=t_vals2,
                    num_samples=int(self.num_fine_samples/2),
                    randomized=randomized,
                )
                viewdirs_enc2 = helper.pos_enc(rays_d_pred, 0, self.deg_view)
                res2 = mlp2( samples2, viewdirs_enc2)

                comp_rgb2, acc2, weights2, _ = helper.volumetric_rendering(
                    res2["rgb"],
                    res2["density"],
                    t_vals2,
                    rays_d_pred,
                    white_bkgd=white_bkgd,
                )
                res2["weights"] = weights2

            comp_rgb =  comp_rgb1 + comp_rgb2 
            acc = acc1 + acc2

            res = {}
            res["comp_rgb"] = comp_rgb
            res["acc"] = acc
            res["normal"] = normal
            ret.append([res,res1,res2])
        ret.append((uvst_pred, uvst_repred))
        # TODO: change to dic
        return ret


@gin.configurable()
class LitSNeRF(LitModel):
    def __init__(
        self,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitSNeRF, self).__init__()
        self.model = SNeRF()

    def setup(self, stage: Optional[str] = None) -> None:
        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.white_bkgd = self.trainer.datamodule.white_bkgd

    def training_step(self, batch, batch_idx):

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        res_coarse = rendered_results[0]
        res_fine = rendered_results[1]
        res_uvst = rendered_results[2]
        target = batch["target"]

        loss0 = helper.img2mse(res_coarse[0]["comp_rgb"], target)
        loss1 = helper.img2mse(res_fine[0]["comp_rgb"], target)

        # loss2 = torch.norm(torch.add(res_uvst[0],res_uvst[1]), p=2)
        view = helper.get_rays_d(res_uvst[0])
        view_pred = helper.get_rays_d(res_uvst[1])
        loss2 = 0.1 * torch.mean((1.0 - torch.sum(view * view_pred, dim=-1)).sum(dim=-1))

        loss_normal_reg_coarse = helper.normal_loss(res_coarse[0]["normal"])
        loss_normal_reg_fine = helper.normal_loss(res_fine[0]["normal"])
        loss_normal_coarse = sum(self.pred_normal_loss(res_coarse[i]["normals"], res_coarse[i]["normals_pred"], res_coarse[i]["weights"]) for i in range(1,3))
        loss_normal_fine = sum(self.pred_normal_loss(res_fine[i]["normals"], res_fine[i]["normals_pred"], res_fine[i]["weights"]) for i in range(1,3))
        loss_normal = 3e-5 * loss_normal_coarse + 3e-4 * loss_normal_fine
        loss_normal_reg = 3e-5 * loss_normal_reg_coarse + 3e-4 * loss_normal_reg_fine
       
        loss = loss1 + loss0 + 0.05*loss2 + loss_normal 
        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)

        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss0", loss0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss1", loss1, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss2", loss2, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss_normal", loss_normal, on_step=True, prog_bar=True, logger=True)
        # self.log("train/loss3", loss3, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)

        return loss

    def render_rays(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far
        )
        rgb_fine = rendered_results[1][0]["comp_rgb"]
        target = batch["target"]
        ret["target"] = target
        ret["rgb"] = rgb_fine
        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = gin.query_parameter("run.max_steps")

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def validation_epoch_end(self, outputs):
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)

            result_path = os.path.join(self.logdir, "results.json")
            self.write_stats(result_path, psnr, ssim, lpips)

        return psnr, ssim, lpips
    

    def pred_normal_loss(self, normal, normal_pred, weight):
        w = weight
        n = normal
        n_pred = normal_pred
        loss = torch.mean((w * (1.0 - torch.sum(n * n_pred, dim=-1))).sum(dim=-1))

        return loss
