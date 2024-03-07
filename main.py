"""
Entry point training and testing relformer
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.RelPoseDataset import RelPoseDataset
from datasets.KNNCameraPoseDataset import KNNCameraPoseDataset
from models.pose_losses import CameraPoseLoss
from os.path import join
from models.relformer.RelFormer import RelFormer, RelFormer2, BrRwlFormer
from models.DeltaNet import DeltaNet, BaselineRPR, DeltaNetEquiv, TDeltaNet, MSDeltaNet
import sys
import torch
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_rotation_6d

def convert_to_quat(rot_repr_type, est_rel_poses, mode_6d):
    batch_size = est_rel_poses.shape[0]
    if rot_repr_type != 'q' and rot_repr_type != '10d':
        if rot_repr_type == '6d':
            rot_repr = est_rel_poses[:, 3:]
            if mode_6d == 0:
                rot_repr = rotation_6d_to_matrix(rot_repr).transpose(1,2)
            else:
                rot_repr = rotation_6d_to_matrix(rot_repr)            
        elif rot_repr_type == '9d':
            # apply SVD orthogonalization to get the rotation matrix
            rot_repr = est_rel_poses[:, 3:]
            rot_repr = rot_repr.reshape(batch_size, 3, 3).transpose(1, 2)
        quaternions = matrix_to_quaternion(rot_repr)
        est_rel_poses = torch.cat((est_rel_poses[:, :3], quaternions), dim=1)
    return est_rel_poses

def load_arch(config, args, device, device_id):
    arch = config.get("arch")
    is_multi_scale = False
    if arch == "deltanet":
        model = DeltaNet(config, args.rpr_backbone_path, args.uncertainty)
        model.to(device)
        # support freeze
        estimate_position_with_prior = config.get("position_with_prior")
        estimate_rotation_with_prior = config.get("rotation_with_prior")
        freeze = False
        if estimate_rotation_with_prior:
            freeze = True
            # exclude rotation-related
            freeze_exclude_phrase = ["_q."] # freeze backbone and all position-related modules
        elif estimate_position_with_prior:
            freeze = True
            # exclude position-related
            freeze_exclude_phrase = ["_x."] # freeze backbone and all rotation-related modules
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                    parameter.requires_grad_(False)
                else:
                    print(name)

    elif arch == "baseline":
        model = BaselineRPR(args.rpr_backbone_path, args.uncertainty).to(device)
    elif arch == "deltanetequiv":
        model = DeltaNetEquiv(config).to(device)
    elif arch == "tdeltanet":
        model = TDeltaNet(config, args.rpr_backbone_path).to(device)
    elif arch == "msdeltanet":
        model = MSDeltaNet(config, args.rpr_backbone_path).to(device)
        is_multi_scale = True
        assert rot_repr_type == '6d' or rot_repr_type == 'q'
    else:
        raise NotImplementedError(arch)
    
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id), strict=False)
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))
        
    return model, is_multi_scale

def run_main(args):
    
    utils.init_logger()    

    # Record execution details
    logging.info("Start {} experiment for RelFormer".format(args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))
    logging.info("Running with command line params:\n{}".format(
        "{}".format(' '.join(sys.argv[:]))))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    if not args.uncertainty:
        torch.manual_seed(torch_seed)
        np.random.seed(numpy_seed)
    if use_cuda:
        if not args.uncertainty:
            torch.backends.cudnn.fdeterministic = True
        torch.backends.cudnn.benchmark = False        
        device_id = 'cuda:' + args.gpu    
    device = torch.device(device_id)

    rot_repr_type = config.get('rot_repr_type')
    if rot_repr_type is not None and rot_repr_type != "q":
        if rot_repr_type == '6d':
            config['rot_dim'] = 6
        elif rot_repr_type == '9d':
            config['rot_dim'] = 9
        elif rot_repr_type == '10d':
            config['rot_dim'] = 4 # we output quaternions
        else:
            raise NotImplementedError(rot_repr_type)
    else:
        config["rot_dim"] = 4
        config["rot_repr_type"] = 'q'
        rot_repr_type = 'q'

    
    model, is_multi_scale = load_arch(config, args, device, device_id)    
    
    eval_reductions = config.get("reduction_eval")
    train_reductions = config.get("reduction")
    mode_6d = args.mode_6d
    is_reproj = config.get("reproj")
    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        transform = utils.train_transforms.get('baseline')
        reproj_transforms = utils.reproj_transforms.get('baseline')
        reproj_transforms_inv = utils.reproj_transforms_inv.get('baseline')
        if args.is_knn:
            train_dataset = KNNCameraPoseDataset(args.dataset_path, args.labels_file, args.refs_file, args.knn_file, transform, args.knn_len, is_reproj=is_reproj)
        else:
            train_dataset = RelPoseDataset(args.dataset_path, args.labels_file, is_reproj, transform, reproj_transforms)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(train_dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        # Resetting temporal loss used for logging
        running_loss = 0.0
        n_samples = 0
        reproj_l2_loss = nn.MSELoss(reduction='sum')
        skip_n_epochs = config.get('skip_n_epochs')

        writer, args.save_dir = utils.init_tensorbaord_log(args, saved_path=args.log_dir)

        i = 0
        criterion1 = torch.zeros([1], dtype=torch.float, device=device)
        criterion = torch.zeros([1], dtype=torch.float, device=device)

        start_reproj_epoch = 0 #n_epochs//30

        for epoch in range(n_epochs):
            # resume checkpoint from same LR
            if epoch < skip_n_epochs:
                for st in range(len(dataloader)):
                    optim.step()
                scheduler.step()
                continue

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_rel_poses = minibatch['rel_pose'].to(dtype=torch.float32)
                gt_rel_poses_orig = gt_rel_poses
                batch_size = gt_rel_poses.shape[0]
                if rot_repr_type != 'q' and rot_repr_type != '10d':
                    q = gt_rel_poses[:, 3:]
                    #rot_mat1 = utils.compute_rotation_matrix_from_quaternion(q)
                    rot_mat = quaternion_to_matrix(q)
                    if rot_repr_type == '6d':
                        # take first two columns and flatten
                        if mode_6d == 0:
                            rot_repr = rot_mat[:,:,:-1].transpose(1,2).reshape(batch_size, -1)
                        else:
                            rot_repr = matrix_to_rotation_6d(rot_mat)
                        #rot_repr = matrix_to_rotation_6d(rot_mat.transpose(1,2))
                        #rot_repr = matrix_to_rotation_6d(rot_mat)
                    elif rot_repr_type == '9d':
                        # GT is the rot matrix
                        rot_repr = rot_mat.transpose(1,2).reshape(batch_size, -1)

                    gt_rel_poses = torch.cat((gt_rel_poses[:, :3], rot_repr), dim=1)
                n_samples += batch_size
                n_total_samples += batch_size

                neighbor_poses = minibatch['ref_pose'].to(device).to(dtype=torch.float32)
                 # Estimate the relative pose
                # Zero the gradients
                optim.zero_grad()
                res = model(minibatch)

                est_rel_poses = res.get("rel_pose")
                est_reproj = res.get("reproj")
                if is_multi_scale:
                    criterion = 0.0
                    for reduction in train_reductions:
                        criterion += pose_loss(est_rel_poses[reduction], gt_rel_poses)
                    criterion /= len(train_reductions)
                else:
                    if 0:#rot_repr_type == '9d':
                        # not supported for multi-scale
                        # apply SVD orthogonalization to get the rotation matrix
                        est_rel_rot = est_rel_poses[:, 3:]
                        est_rel_rot = est_rel_rot.transpose(1,2).reshape(batch_size, -1)
                        est_rel_poses = torch.cat((est_rel_poses[:, :3], est_rel_rot), dim=1)

                    criterion = pose_loss(est_rel_poses, gt_rel_poses)

                if is_reproj == 1:
                    gt_reproj = minibatch['reproj']
                    non_zero_mask = torch.where(gt_reproj > 0, 1, 0)
                    criterion1 = reproj_l2_loss(est_reproj*non_zero_mask, gt_reproj)
                    criterion += args.reproj_loss_w * criterion1
                                    
                if is_reproj == 2 and epoch >= start_reproj_epoch:
                    gt_reproj_orig = minibatch.get('reproj_orig')
                    non_zero_mask = torch.where(gt_reproj_orig > 0, 1, 0)
                    #est_rel_poses = gt_rel_poses
                    rot_repr = est_rel_poses[:, 3:]
                    if rot_repr_type == '6d':                    
                        if mode_6d == 0:
                            rot_repr = rotation_6d_to_matrix(rot_repr).transpose(1,2)
                        else:
                            rot_repr = rotation_6d_to_matrix(rot_repr)                        
                        q = matrix_to_quaternion(rot_repr)
                    elif rot_repr_type == '9d':
                        # apply SVD orthogonalization to get the rotation matrix                        
                        rot_repr = rot_repr.reshape(batch_size, 3, 3).transpose(1,2)
                        q = matrix_to_quaternion(rot_repr)
                    else:                        
                        q = rot_repr
                    est_rel_poses1 = torch.cat((est_rel_poses[:, :3], q), dim=1)
                    ref_pose = utils.compute_abs_pose_torch(est_rel_poses1, minibatch.get('query_pose').float())
                    reproj_img = utils.reproject_RGB(minibatch.get('query_orig'), minibatch.get('depth'), minibatch.get('query_pose'), ref_pose, args.is_knn)
                    reproj_img = reproj_img.clamp(0, 1)
                    cnt_non_zero = non_zero_mask.sum()
                    criterion1 = reproj_l2_loss(reproj_img * non_zero_mask, gt_reproj_orig)/cnt_non_zero
                    criterion += args.reproj_loss_w * criterion1
                    if i%100 == 0:
                        utils.log_img_to_tensorboard_triplet(writer, gt_reproj_orig, reproj_img, step=i)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    if is_multi_scale:
                        est_rel_poses = est_rel_poses[train_reductions[0]]  # for printing purposes
                    if rot_repr_type != 'q' and rot_repr_type != '10d':
                        if rot_repr_type == '6d':
                            rot_repr = est_rel_poses[:, 3:]
                            if mode_6d == 0:
                                rot_repr = rotation_6d_to_matrix(rot_repr).transpose(1, 2)
                            else:
                                rot_repr = rotation_6d_to_matrix(rot_repr)
                        elif rot_repr_type == '9d':
                            rot_repr = est_rel_poses[:, 3:].reshape(batch_size, 3, 3).transpose(1,2)
                        q = matrix_to_quaternion(rot_repr)
                        est_rel_poses = torch.cat((est_rel_poses[:,:3], q), dim=1)

                    posit_err, orient_err = utils.pose_err(est_rel_poses.detach(), gt_rel_poses_orig.detach())
                    msg = "[Batch-{}/Epoch-{}] running relative camera pose loss: {:.3f}, camera pose error: {:.2f}[m], {:.2f}[deg] reproj loss {:.2f}".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_freq_print),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item(),
                                                                        criterion1.mean().item())
                    posit_err, orient_err = utils.pose_err(neighbor_poses.detach(), minibatch['query_pose'].to(dtype=torch.float32).detach())
                    msg = msg + ", distance from neighbor images: {:.2f}[m], {:.2f}[deg]".format(posit_err.mean().item(),
                                                                         orient_err.mean().item())
                    logging.info(msg)

                    utils.log_to_tensorboard(writer, criterion.item(), i, criterion1.item())

                    # Resetting temporal loss used for logging
                    running_loss = 0.0
                    n_samples = 0
                    i += 1

            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_relformer_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_relformer_final.pth')

    else: # Test
        # Set to eval mode
        model.eval()

        reproj_l2_loss = nn.MSELoss(reduction='sum')

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        is_reproj = 0
        if args.is_knn:
            test_dataset = KNNCameraPoseDataset(args.dataset_path, args.test_labels_file, args.refs_file, args.test_knn_file, transform, args.knn_len, args.knn_len, False)
        else:
            test_dataset = RelPoseDataset(args.dataset_path, args.test_labels_file, is_reproj, transform, transform, False)

        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)
        pose_stats = np.zeros((len(dataloader.dataset), 3))
        criterion1 = 0        
        
        if args.uncertainty:        
            num_of_runs = 40
            num_of_rows = dataloader.dataset.__len__()
            pose_list = np.zeros((num_of_rows, num_of_runs, 7))       
        else:
            num_of_runs = 1
            
        for j in range(num_of_runs):            
                        
            if args.uncertainty:            
                model, is_multi_scale = load_arch(config, args, device, device_id)    
            
            with torch.no_grad():
                for i, minibatch in enumerate(dataloader, 0):
                    for k, v in minibatch.items():
                        minibatch[k] = v.to(device)
                    gt_rel_pose = minibatch.get('rel_pose').to(dtype=torch.float32)

                    # Forward pass to predict the initial pose guess
                    if args.is_knn:
                        posit_err = orient_err = 0
                        for j in range(args.knn_len):
                            if args.knn_len == 1:
                                gt_rel_pose_i = gt_rel_pose
                                ref_i = minibatch.get('ref')
                            else:
                                gt_rel_pose_i = gt_rel_pose[:,j,:]
                                ref_i = minibatch.get('ref')[:,j,:]
                            minibatch_i = {'query': minibatch.get('query'), 'ref': ref_i}                        
                            t0 = time.time()
                            res = model(minibatch_i)
                            est_rel_pose = res['rel_pose']
                            torch.cuda.synchronize()
                            tn = time.time()
                            est_rel_pose = convert_to_quat(rot_repr_type, est_rel_pose, mode_6d)
                            # Evaluate error
                            posit_err_i, orient_err_i = utils.pose_err(est_rel_pose, gt_rel_pose_i)
                            posit_err += posit_err_i.squeeze()
                            orient_err += orient_err_i.squeeze()

                        posit_err /= args.knn_len
                        orient_err /= args.knn_len
                            
                    else:
                        t0 = time.time()
                        res = model(minibatch)
                        est_rel_pose = res['rel_pose']
                        torch.cuda.synchronize()
                        tn = time.time()

                        if is_multi_scale:
                            posit_err = 0.0
                            orient_err = 0.0
                            for reduction in eval_reductions:
                                # Evaluate error
                                curr_posit_err, curr_orient_err = utils.pose_err(convert_to_quat(rot_repr_type,
                                                                                                est_rel_pose[reduction]),
                                                                                gt_rel_pose, mode_6d)
                                posit_err += curr_posit_err
                                orient_err += curr_orient_err

                            posit_err /= len(eval_reductions)
                            orient_err /= len(eval_reductions)

                        else:
                            est_rel_pose = convert_to_quat(rot_repr_type, est_rel_pose, mode_6d)
                            # Evaluate error
                            posit_err, orient_err = utils.pose_err(est_rel_pose, gt_rel_pose)

                        
                    # Collect statistics
                    pose_stats[i, 0] = posit_err.item()
                    pose_stats[i, 1] = orient_err.item()
                    pose_stats[i, 2] = (tn - t0)*1000
                    
                    #if i==num_of_rows:
                    #    break       
                    
                    if args.uncertainty:                    
                        pose_list[i, j] = est_rel_pose.squeeze().cpu().numpy()

                    msg = "Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms], reproj_err: {:2f}".format(
                        pose_stats[i, 0],  pose_stats[i, 1],  pose_stats[i, 2], criterion1)

                    #posit_err, orient_err = utils.pose_err(minibatch['ref_pose'].to(device).to(dtype=torch.float32).detach(),
                    #                                       minibatch['query_pose'].to(dtype=torch.float32).detach())
                    #msg = msg + ", distance from neighbor images: {:.2f}[m], {:.2f}[deg]".format(posit_err.mean().item(),
                    #                                                                             orient_err.mean().item())
                    logging.info(msg)                    
                               

            # Record overall statistics
            logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
            logging.info("Median pose error: {:.3f}, {:.3f}".format(np.nanmedian(pose_stats[:, 0]), np.nanmedian(pose_stats[:, 1])))
            logging.info("Mean RPR inference time:{:.2f}[ms]".format(np.mean(pose_stats)))
            
        if args.uncertainty:
            np.savetxt(args.uncertainty_file, pose_list.reshape(num_of_rows, -1), delimiter=",", fmt='%s')
                    posit_err /= args.knn_len
                    orient_err /= args.knn_len
                        
                else:
                    t0 = time.time()
                    res = model(minibatch)
                    est_rel_pose = res['rel_pose']
                    #torch.cuda.synchronize()
                    tn = time.time()

                    if is_multi_scale:
                        posit_err = 0.0
                        orient_err = 0.0
                        for reduction in eval_reductions:
                            # Evaluate error
                            curr_posit_err, curr_orient_err = utils.pose_err(convert_to_quat(rot_repr_type,
                                                                                            est_rel_pose[reduction]),
                                                                            gt_rel_pose, mode_6d)
                            posit_err += curr_posit_err
                            orient_err += curr_orient_err

                        posit_err /= len(eval_reductions)
                        orient_err /= len(eval_reductions)

                    else:
                        est_rel_pose = convert_to_quat(rot_repr_type, est_rel_pose, mode_6d)
                        # Evaluate error
                        posit_err, orient_err = utils.pose_err(est_rel_pose, gt_rel_pose)

                
                # Collect statistics
                pose_stats[i, 0] = posit_err.item()
                pose_stats[i, 1] = orient_err.item()
                pose_stats[i, 2] = (tn - t0)*1000

                msg = "Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms], reproj_err: {:2f}".format(
                    pose_stats[i, 0],  pose_stats[i, 1],  pose_stats[i, 2], criterion1)

                logging.info(msg)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", help="train or eval", default='test')
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="F:/7Scenes/")
    #arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="F:/CambridgeLandmarks/")
    arg_parser.add_argument("--rpr_backbone_path", help="path to the backbone path", default="models/backbones/efficient-net-b0.pth")
    arg_parser.add_argument("--labels_file", help="pairs file", default="datasets/7Scenes/7scenes_training_pairs.csv")
    #arg_parser.add_argument("--labels_file", help="pairs file", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv")
    arg_parser.add_argument("--refs_file", help="path to a file mapping reference images to their poses", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv")
    arg_parser.add_argument("--knn_file", help="path to a file mapping query images to their knns", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv_with_netvlads.csv-knn-cambridge_four_scenes.csv_with_netvlads_orig.csv")
    #arg_parser.add_argument("--test_knn_file", help="path to a file mapping query images to their knns", default="datasets/7Scenes_knn/7scenes_knn_test_neigh_chess.csv")
    arg_parser.add_argument("--test_knn_file", help="path to a file mapping query images to their knns", default="datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_OldHospital_test.csv_with_netvlads.csv-knn-cambridge_four_scenes.csv_with_netvlads.csv")
    arg_parser.add_argument("--test_labels_file", help="pairs file", default="datasets/7Scenes_test_NN/NN_7scenes_fire.csv")    
    #arg_parser.add_argument("--test_labels_file", help="pairs file", default="datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_OldHospital_test.csv")    
    arg_parser.add_argument("--config_file", help="path to configuration file", default="config/7scenes_config.json")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained RPR model")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained RPR model", default="checkpoints_7/relformer_DeltanetEnc_6d_nofire.pth")
    arg_parser.add_argument("--test_dataset_id", default="7scenes", help="test set id for testing on all scenes, options: 7scene OR cambridge")
    #arg_parser.add_argument("--test_dataset_id", default="Cambridge", help="test set id for testing on all scenes, options: 7scene OR cambridge")
    arg_parser.add_argument("--knn_len", help="knn_len", type=int, default="1")
    arg_parser.add_argument("--is_knn", help="is_knn", type=int, default="0")
    arg_parser.add_argument("--gpu", help="gpu id", default="0")
    arg_parser.add_argument("--log_dir", help="log dir", default="logs")
    arg_parser.add_argument("--reproj_loss_w", help="reproj loss weight", type=float, default="1")
    arg_parser.add_argument("--mode_6d", help="mode_6d", type=int, default="0")
    arg_parser.add_argument("--uncertainty", help="uncertainty", type=int, default="0")   
    arg_parser.add_argument("--uncertainty_file", help="uncertainty file", default="pose_results.csv")   
     

    args = arg_parser.parse_args()
    
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    run_main(args)
