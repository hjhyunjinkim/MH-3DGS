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

import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import math
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim, l1_loss_map
from gaussian_renderer import render, network_gui, projection
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def schedule_cameras_for_multiview(all_cameras, camera_usage, camera_coverage, 
                                   iteration, strategy='round_robin', 
                                   subset_size=4):
    if strategy == 'round_robin':
        return round_robin_cameras(all_cameras, iteration, subset_size)
    elif strategy == 'adaptive_coverage':
        return adaptive_coverage_cameras(all_cameras, camera_usage, camera_coverage, subset_size)
    else:
        import random
        return random.sample(all_cameras, subset_size)


def round_robin_cameras(all_cameras, iteration, subset_size=4):
    n = len(all_cameras)
    start_idx = (iteration * subset_size) % n
    chosen = []
    for i in range(subset_size):
        idx = (start_idx + i) % n
        chosen.append(all_cameras[idx])
    return chosen

def adaptive_coverage_cameras(all_cameras, camera_usage, camera_coverage, subset_size=4):
    """
    Compute a score for each camera based on its coverage and usage.
    Here, a camera with a high error (coverage value) and low usage is prioritized.
    Score = camera_coverage[cam.uid] / (1.0 + camera_usage[cam.uid])
    """
    scored_cams = [
        (cam, camera_coverage[cam.uid] / (1.0 + camera_usage[cam.uid]))
        for cam in all_cameras
    ]
    scored_cams.sort(key=lambda x: x[1], reverse=True)
    chosen = [cam for cam, score in scored_cams[:subset_size]]
    return chosen    

def accumulate_multiview_errors(scene, gaussians, pipe, background, cameras, 
                                fused_ssim_available, train_test_exp=False, device='cuda'):
    aggregated_ssim_map = None
    aggregated_l1_map = None
    count = 0
    for cam in cameras:
        render_pkg = render(cam, gaussians, pipe, background, 
                            use_trained_exp=train_test_exp, 
                            separate_sh=False)
        rendered_scene = render_pkg["render"]  
        gt_image = cam.original_image.to(device)
        # if fused_ssim_available:
        #     ssim_map = fused_ssim(rendered_scene.unsqueeze(0), gt_image.unsqueeze(0),
        #                           size_average=False).squeeze(0)  
        # else:
        ssim_map = ssim(rendered_scene, gt_image, size_average=False)
        l1_map = l1_loss_map(rendered_scene, gt_image)
        if aggregated_ssim_map is None:
            aggregated_ssim_map = ssim_map.clone()
            aggregated_l1_map = l1_map.clone()
        else:
            aggregated_ssim_map += ssim_map
            aggregated_l1_map += l1_map
        count += 1
    aggregated_ssim_map /= float(count)
    aggregated_l1_map   /= float(count)
    return aggregated_ssim_map, aggregated_l1_map

def get_scene_extent(xyz_tensor):
    # xyz_tensor: (N,3) of points in the scene
    min_coords = xyz_tensor.min(dim=0).values
    max_coords = xyz_tensor.max(dim=0).values
    # An overall size of the bounding box:
    extent_vec = max_coords - min_coords
    bounding_size = extent_vec.norm().item()
    return bounding_size

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    num_points_per_iteration = []

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    camera_number = len(scene.getTrainCameras().copy())

    all_cameras = scene.getTrainCameras()  # list of your camera objects
    camera_usage = {cam.uid: 0 for cam in all_cameras}  # track usage count
    camera_coverage = {cam.uid: 0 for cam in all_cameras}  # track coverage or "problem" metric

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        projection_pkg = projection(viewpoint_cam, gaussians.get_xyz)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if opt.lambda_opacity_reg > 0 or opt.lambda_scale_reg > 0:
            alpha_val = gaussians.get_opacity
            scale_val = gaussians.get_scaling
            
            opacity_pen = alpha_val.mean()
            scale_pen = scale_val.mean()

            reg_loss = opt.lambda_opacity_reg * opacity_pen + opt.lambda_scale_reg * scale_pen
            loss += reg_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # Densification
            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                densification_progress = (iteration - opt.densify_from_iter) / (opt.densify_until_iter - opt.densify_from_iter)
                current_subset_size = max(int(len(all_cameras) * (1-densification_progress)),1)

                chosen_cams = schedule_cameras_for_multiview(
                    all_cameras, camera_usage, camera_coverage, 
                    iteration=iteration,
                    strategy='round_robin', 
                    subset_size=current_subset_size
                )

                dynamic_coarse_proposal_scale = opt.coarse_max * (1 - densification_progress) + opt.coarse_min * densification_progress
                dynamic_fine_proposal_scale   = opt.fine_max * (1 - densification_progress) + opt.fine_min * densification_progress
                adjusted_voxel_size           = (1 - densification_progress) * opt.voxel_max + densification_progress * opt.voxel_min
                adjusted_coarse_batch_size    = max(1, int((1 - densification_progress) * opt.coarse_batch_size))   
                adjusted_fine_batch_size      = max(1, int((1 - densification_progress) * opt.fine_batch_size)) 

                aggregated_ssim_map, aggregated_l1_map = accumulate_multiview_errors(scene, gaussians, pipe, background, chosen_cams, FUSED_SSIM_AVAILABLE, train_test_exp=dataset.train_test_exp)

                gaussians.combined_mcmc_and_reloc_multiview(
                    ssim_map=aggregated_ssim_map,
                    l1_map=aggregated_l1_map,
                    iteration=iteration,
                    densify_from_iter=opt.densify_from_iter,
                    densify_until_iter=opt.densify_until_iter,
                    voxel_size=adjusted_voxel_size,  
                    alpha=5.0,
                    coarse_batch_size=adjusted_coarse_batch_size,
                    fine_batch_size=adjusted_fine_batch_size,
                    coarse_proposal_scale=dynamic_coarse_proposal_scale,
                    fine_proposal_scale=dynamic_fine_proposal_scale,
                    opacity_threshold=0.005
                )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            num_points = gaussians.get_xyz.shape[0]
            num_points_per_iteration.append(num_points)

        plot_iteration = [30_000]

        if (iteration in plot_iteration):
            print(f"Iteration {iteration} - Num points: {num_points_per_iteration[-1]}") 

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")