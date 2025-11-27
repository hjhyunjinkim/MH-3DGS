#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import ssim, l1_loss_map
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.reloc_utils import compute_relocation_cuda

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

import math
from collections import defaultdict

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def calculate_gradients(self):
        grads = self.xyz_gradient_accum / (self.denom + 1e-8)
        grads[grads.isnan()] = 0.0

        return grads

    def sample_scores_from_maps(self, ssim_map, l1_map):
        # Add a batch dimension if necessary
        if ssim_map.dim() == 3:
            ssim_map = ssim_map.unsqueeze(0)  # [1, channels, height, width]
        if l1_map.dim() == 3:
            l1_map = l1_map.unsqueeze(0)

        # Ensure Gaussian coordinates are within image bounds
        height, width = ssim_map.shape[-2], ssim_map.shape[-1]
        coords_x = (self.get_xyz[:, 0] * (width - 1)).long()
        coords_y = (self.get_xyz[:, 1] * (height - 1)).long()
        coords_x = coords_x.clamp(0, width - 1)
        coords_y = coords_y.clamp(0, height - 1)

        # Sample SSIM and L1 scores from the maps
        ssim_scores = ssim_map[0, :, coords_y, coords_x].mean(dim=0)  
        l1_scores = l1_map[0, :, coords_y, coords_x].mean(dim=0)      

        return ssim_scores, l1_scores

    def point_to_voxel_id(self, point, voxel_size):
        """
        point: (3,) in world coords
        voxel_size: float (or a 3D vector if you want different resolution per axis)
        Returns an (ix, iy, iz) integer tuple or a hashed ID for the point's voxel.
        """
        # For simplicity, assume voxel_size is a single float
        ix = math.floor(point[0].item() / voxel_size)
        iy = math.floor(point[1].item() / voxel_size)
        iz = math.floor(point[2].item() / voxel_size)
        return (ix, iy, iz)

    def compute_voxel_counts(self, xyz, voxel_size):
        """
        xyz: (N,3) tensor of all existing points
        Returns a dict: voxel_id -> count of how many points fall into that voxel.
        """
        voxel_counts = defaultdict(int)
        xyz_cpu = xyz.detach().cpu().numpy()  # move to CPU for a simple loop
        for i in range(xyz_cpu.shape[0]):
            vx = self.point_to_voxel_id(xyz_cpu[i], voxel_size)
            voxel_counts[vx] += 1
        return voxel_counts

    def voxel_density_factor(self, proposal_xyz, voxel_counts, voxel_size, alpha=5.0):
        """
        For each new 'proposal' point, compute a penalty factor in [0,1].
        factor = 1.0 / (1.0 + alpha * voxel_count)
        The higher the local voxel_count, the lower the factor.
        """
        factors = []
        xyz_cpu = proposal_xyz.detach().cpu().numpy()
        for i in range(xyz_cpu.shape[0]):
            vx = self.point_to_voxel_id(xyz_cpu[i], voxel_size)
            c = voxel_counts.get(vx, 0)
            f = 1.0 / (1.0 + alpha * c)
            factors.append(f)
        return torch.tensor(factors, dtype=torch.float, device=proposal_xyz.device)

    def importance_scores(self, ssim, Ll1, opacity_weight=0.8, ssim_weight=0.5, lli_weight=0.5):    #0.5 
        def robust_normalize(x):
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + 1e-6)

        opacity_norm = robust_normalize(self.get_opacity.squeeze())
        ssim_norm = robust_normalize(ssim)
        lli_norm = robust_normalize(Ll1)

        importance_scores = (
            opacity_weight * opacity_norm +
            ssim_weight * ssim_norm +
            lli_weight * lli_norm
        )

        importance_scores = torch.sigmoid(importance_scores)
        return importance_scores

    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1),
                                max=1.0 - torch.finfo(torch.float32).eps,
                                min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))
        return (self._xyz[idxs],
                self._features_dc[idxs],
                self._features_rest[idxs],
                new_opacity,
                new_scaling,
                self._rotation[idxs])

    def _sample_alives(self, probs, num, alive_indices=None):
        # Normalize probabilities and sample indices.
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        # Ensure ratio has the proper length.
        ratio = torch.bincount(sampled_idxs, minlength=self._xyz.shape[0]).unsqueeze(-1)
        return sampled_idxs, ratio

    def replace_tensors_to_optimizer(self, inds=None):
        # This function resets optimizer state for selected indices.
        tensors_dict = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation
        }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        torch.cuda.empty_cache()
        return optimizable_tensors

    def combined_mcmc_and_reloc_multiview(
        self,
        ssim_map,
        l1_map,
        iteration,
        densify_from_iter=500,
        densify_until_iter=15000,
        voxel_size=0.01,
        alpha=6.0,
        coarse_batch_size=1500,
        fine_batch_size=8000,
        coarse_proposal_scale=10.0,
        fine_proposal_scale=2.0,
        opacity_threshold=0.005
    ):
        # ---------------------
        # Step 1: Relocate dead points first.
        # ---------------------
        dead_mask = (self.get_opacity <= opacity_threshold).squeeze(-1)
        if dead_mask.sum() > 0:
            alive_mask = ~dead_mask
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            alive_indices = alive_mask.nonzero(as_tuple=True)[0]
            if alive_indices.numel() > 0:

                probs = self.get_opacity[alive_indices, 0]
                reinit_idx, ratio = self._sample_alives(probs, num=dead_indices.shape[0], alive_indices=alive_indices)

                updated = self._update_params(reinit_idx, ratio=ratio)
                self._xyz[dead_indices] = updated[0]
                self._features_dc[dead_indices] = updated[1]
                self._features_rest[dead_indices] = updated[2]
                self._opacity[dead_indices] = updated[3]
                self._scaling[dead_indices] = updated[4]
                self._rotation[dead_indices] = updated[5]
                self._opacity[reinit_idx] = self._opacity[dead_indices]
                self._scaling[reinit_idx] = self._scaling[dead_indices]
                self.replace_tensors_to_optimizer(inds=reinit_idx)

        # ---------------------
        # Step 2: Densification (only if within the densification phase)
        # ---------------------
        if densify_from_iter <= iteration <= densify_until_iter:
            # Sample error scores from SSIM and L1 maps.
            ssim_scores, l1_scores = self.sample_scores_from_maps(ssim_map, l1_map)
            error_importance = self.importance_scores(ssim_scores, l1_scores)
            error_importance = torch.clamp(error_importance, min=0)
            voxel_counts = self.compute_voxel_counts(self.get_xyz, voxel_size)

            # --- Coarse proposals ---
            selected_idx_coarse = torch.multinomial(error_importance, coarse_batch_size, replacement=True)
            selected_idx_coarse = torch.unique(selected_idx_coarse)
            density_factor_coarse = 1 / (1 + torch.log(1 + self.denom[selected_idx_coarse]))
            random_offset_coarse = torch.normal(
                mean=0.0,
                std=(coarse_proposal_scale *
                    self.get_scaling[selected_idx_coarse] *
                    density_factor_coarse).clamp(min=1e-6)
            )
            proposal_points_coarse = self.get_xyz[selected_idx_coarse] + random_offset_coarse
            density_factor_voxel_coarse = self.voxel_density_factor(proposal_points_coarse,
                                                                    voxel_counts,
                                                                    voxel_size,
                                                                    alpha)
            proposal_importance_coarse = error_importance[selected_idx_coarse]
            acceptance_prob_coarse = torch.sigmoid(proposal_importance_coarse - 0.5) * density_factor_voxel_coarse
            accepted_mask_coarse = (torch.rand_like(acceptance_prob_coarse) < acceptance_prob_coarse)
            new_points_coarse = proposal_points_coarse[accepted_mask_coarse]
            new_features_dc_coarse = self._features_dc[selected_idx_coarse][accepted_mask_coarse]
            new_features_rest_coarse = self._features_rest[selected_idx_coarse][accepted_mask_coarse]
            new_opacity_coarse = self._opacity[selected_idx_coarse][accepted_mask_coarse]
            new_scaling_coarse = self._scaling[selected_idx_coarse][accepted_mask_coarse]
            new_rotation_coarse = self._rotation[selected_idx_coarse][accepted_mask_coarse]

            # --- Fine proposals ---
            selected_idx_fine = torch.multinomial(error_importance, fine_batch_size, replacement=True)
            selected_idx_fine = torch.unique(selected_idx_fine)
            density_factor_fine = 1 / (1 + torch.log(1 + self.denom[selected_idx_fine]))
            random_offset_fine = torch.normal(
                mean=0.0,
                std=(fine_proposal_scale *
                    self.get_scaling[selected_idx_fine] *
                    density_factor_fine).clamp(min=1e-6)
            )
            proposal_points_fine = self.get_xyz[selected_idx_fine] + random_offset_fine
            density_factor_voxel_fine = self.voxel_density_factor(proposal_points_fine,
                                                                voxel_counts,
                                                                voxel_size,
                                                                alpha)
            proposal_importance_fine = error_importance[selected_idx_fine]
            acceptance_prob_fine = torch.sigmoid(proposal_importance_fine - 0.5) * density_factor_voxel_fine
            accepted_mask_fine = (torch.rand_like(acceptance_prob_fine) < acceptance_prob_fine)
            new_points_fine = proposal_points_fine[accepted_mask_fine]
            new_features_dc_fine = self._features_dc[selected_idx_fine][accepted_mask_fine]
            new_features_rest_fine = self._features_rest[selected_idx_fine][accepted_mask_fine]
            new_opacity_fine = self._opacity[selected_idx_fine][accepted_mask_fine]
            new_scaling_fine = self._scaling[selected_idx_fine][accepted_mask_fine]
            new_rotation_fine = self._rotation[selected_idx_fine][accepted_mask_fine]

            # Combine accepted proposals from coarse and fine
            accepted_xyz = torch.cat([new_points_coarse, new_points_fine], dim=0)
            accepted_features_dc = torch.cat([new_features_dc_coarse, new_features_dc_fine], dim=0)
            accepted_features_rest = torch.cat([new_features_rest_coarse, new_features_rest_fine], dim=0)
            accepted_opacity = torch.cat([new_opacity_coarse, new_opacity_fine], dim=0)
            accepted_scaling = torch.cat([new_scaling_coarse, new_scaling_fine], dim=0)
            accepted_rotation = torch.cat([new_rotation_coarse, new_rotation_fine], dim=0)

            if accepted_xyz.shape[0] > 0:
                self.densification_postfix(
                    accepted_xyz,
                    accepted_features_dc,
                    accepted_features_rest,
                    accepted_opacity,
                    accepted_scaling,
                    accepted_rotation
                )
