"""
Entry point training and testing iAPR
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
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d, rotation_6d_to_matrix
def convert_to_quat(rot_repr_type, est_rel_poses):
    batch_size = est_rel_poses.shape[0]
    if rot_repr_type != 'q' and rot_repr_type != '10d':
        if rot_repr_type == '6d':
            rot_repr = est_rel_poses[:, 3:]
            #rot_repr = utils.compute_rotation_matrix_from_ortho6d(rot_repr)
            rot_repr = rotation_6d_to_matrix(rot_repr).transpose(1,2)
        elif rot_repr_type == '9d':
            # apply SVD orthogonalization to get the rotation matrix
            rot_repr = est_rel_poses[:, 3:]
            #rot_repr = utils.symmetric_orthogonalization(rot_repr)
            rot_repr = rot_repr.reshape(batch_size, 3, 3).transpose(1, 2)
        #quaternions = utils.compute_quaternions_from_rotation_matrices(rot_repr)
        quaternions = matrix_to_quaternion(rot_repr)
        est_rel_poses = torch.cat((est_rel_poses[:, :3], quaternions), dim=1)
    return est_rel_poses

def get_knn_indices(query, db):
    distances = torch.linalg.norm(db-query, axis=1)
    return torch.argsort(distances)

if __name__ == "__main__":
    utils.set_proxy()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", help="train or eval", default='train')
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/data/datasets/7Scenes/")
    #arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/data/datasets/CAMBRIDGE_dataset/")
    arg_parser.add_argument("--rpr_backbone_path", help="path to the backbone path", default="models/backbones/efficient-net-b0.pth")
    arg_parser.add_argument("--labels_file", help="pairs file", default="datasets/7Scenes/7scenes_training_pairs.csv")
    #arg_parser.add_argument("--labels_file", help="pairs file", default="datasets/7Scenes_no/7scenes_training_pairs_no_fire.csv")
    #arg_parser.add_argument("--labels_file", help="pairs file", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv")
    #arg_parser.add_argument("--labels_file", help="pairs file", default="datasets/CambridgeLandmarks/cambridge_training_pairs_visloc.csv")
    arg_parser.add_argument("--refs_file", help="path to a file mapping reference images to their poses", default="datasets/7Scenes_knn/7scenes_all_scenes.csv")
    #arg_parser.add_argument("--refs_file", help="path to a file mapping reference images to their poses", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv")
    #arg_parser.add_argument("--knn_file", help="path to a file mapping query images to their knns", default="datasets/CambridgeLandmarks/cambridge_training_pairs_20r.csv")
    arg_parser.add_argument("--knn_file", help="path to a file mapping query images to their knns", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv_with_netvlads.csv-knn-cambridge_four_scenes.csv_with_netvlads_orig.csv")
    #arg_parser.add_argument("--knn_file", help="path to a file mapping query images to their knns", default="datasets/CambridgeLandmarks/CambridgeLandmarks_knn_train.csv")
    #arg_parser.add_argument("--test_knn_file", help="path to a file mapping query images to their knns", default="datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_StMarysChurch_test.csv_with_netvlads.csv-knn-cambridge_four_scenes.csv_with_netvlads.csv")
    #arg_parser.add_argument("--test_knn_file", help="path to a file mapping query images to their knns", default="datasets/7Scenes_test_pairs/7scenes_knn_test_neigh_chess.csv")
    arg_parser.add_argument("--test_knn_file", help="path to a file mapping query images to their knns", default="datasets/7Scenes_knn/7scenes_knn_test_neigh_chess.csv")
    arg_parser.add_argument("--test_labels_file", help="pairs file", default="datasets/7Scenes_test_NN/NN_7scenes_chess.csv")
    #arg_parser.add_argument("--test_labels_file", help="pairs file", default="datasets/7Scenes_knn/abs_7scenes_pose.csv_chess_test.csv")
    #arg_parser.add_argument("--test_labels_file", help="pairs file", default="datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_StMarysChurch_test.csv")
    arg_parser.add_argument("--config_file", help="path to configuration file", default="config/7scenes_config.json")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained RPR model")
    arg_parser.add_argument("--test_dataset_id", default="7scenes", help="test set id for testing on all scenes, options: 7scene OR cambridge")
    #arg_parser.add_argument("--test_dataset_id", default="Cambridge", help="test set id for testing on all scenes, options: 7scene OR cambridge")
    arg_parser.add_argument("--knn_len", help="knn_len", type=int, default="1")
    arg_parser.add_argument("--is_knn", help="is_knn", type=int, default="0")
    arg_parser.add_argument("--gpu", help="gpu id", default="0")
    arg_parser.add_argument("--log_dir", help="log dir", default="logs")
    arg_parser.add_argument("--reproj_loss_w", help="reproj loss weight", type=float, default="1")

    args = arg_parser.parse_args()
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
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.fdeterministic = True
        torch.backends.cudnn.benchmark = False
        #device_id = config.get('device_id')
        device_id = 'cuda:' + args.gpu
    np.random.seed(numpy_seed)
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

    arch = config.get("arch")
    is_multi_scale = False
    if arch == "relformer2":
        model = RelFormer2(config, args.rpr_backbone_path).to(device)
    elif arch == "relformer":
        model = RelFormer(config, args.rpr_backbone_path).to(device)
    elif arch == "b-relformer":
        model = BrRwlFormer(config, args.rpr_backbone_path).to(device)
    elif arch == "deltanet":
        model = DeltaNet(config, args.rpr_backbone_path)
        #model = torch.nn.DataParallel(model, device_ids=[1, 2, 3, 4])
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
        model = BaselineRPR(args.rpr_backbone_path).to(device)
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
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id), strict=True)
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    eval_reductions = config.get("reduction_eval")
    train_reductions = config.get("reduction")
    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        train_w_triplet_loss = False # read from config #TODO Triplet loss

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
        is_reproj = config.get("reproj")
        if args.is_knn:
            train_dataset = KNNCameraPoseDataset(args.dataset_path, args.labels_file, args.refs_file, args.knn_file, transform, args.knn_len)
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
                if is_reproj:
                    gt_reproj = minibatch['reproj']
                gt_rel_poses_orig = gt_rel_poses
                batch_size = gt_rel_poses.shape[0]
                if rot_repr_type != 'q' and rot_repr_type != '10d':
                    q = gt_rel_poses[:, 3:]
                    #rot_mat1 = utils.compute_rotation_matrix_from_quaternion(q)
                    rot_mat = quaternion_to_matrix(q)
                    if rot_repr_type == '6d':
                        # take first two columns and flatten
                        rot_repr = rot_mat[:,:,:-1].transpose(1,2).reshape(batch_size, -1)
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
                        est_rel_rot = utils.symmetric_orthogonalization(est_rel_poses[:, 3:])
                        est_rel_rot = est_rel_rot.transpose(1,2).reshape(batch_size, -1)
                        est_rel_poses = torch.cat((est_rel_poses[:, :3], est_rel_rot), dim=1)

                    criterion = pose_loss(est_rel_poses, gt_rel_poses)

                if is_reproj == 1:
                    non_zero_mask = torch.where(gt_reproj > 0, 1, 0)
                    criterion1 = reproj_l2_loss(est_reproj*non_zero_mask, gt_reproj)
                    criterion += args.reproj_loss_w * criterion1
                    #torchvision.utils.save_image(gt_reproj, 'out_reproj/gt.png')
                    #torchvision.utils.save_image(est_reproj*zero_mask, 'out_reproj/out.png')
                   # break
                if is_reproj == 2:# and epoch > 20:
                    gt_reproj_orig = minibatch.get('reproj_orig')
                    non_zero_mask = torch.where(gt_reproj_orig > 0, 1, 0)
                    rot_repr = est_rel_poses[:, 3:]
                    if rot_repr_type == '6d':
                        #rot_repr = utils.compute_rotation_matrix_from_ortho6d(rot_repr)
                        rot_repr = rotation_6d_to_matrix(rot_repr).transpose(1,2)
                        #q = utils.compute_quaternions_from_rotation_matrices(rot_repr)
                        q = matrix_to_quaternion(rot_repr)
                    elif rot_repr_type == '9d':
                        # apply SVD orthogonalization to get the rotation matrix
                        #rot_repr = utils.symmetric_orthogonalization(rot_repr)
                        rot_repr = rot_repr.reshape(batch_size, 3, 3).transpose(1,2)
                        q = matrix_to_quaternion(rot_repr)
                    else:                        
                        q = rot_repr
                    est_rel_poses1 = torch.cat((est_rel_poses[:, :3], q), dim=1)
                    #rot_repr = gt_rel_poses[:, 3:]
                    #rot_repr = utils.compute_rotation_matrix_from_ortho6d(rot_repr)
                    #rot_repr = rotation_6d_to_matrix(rot_repr)
                    #q = utils.compute_quaternions_from_rotation_matrices(rot_repr)
                    #q = matrix_to_quaternion(rot_repr)
                    #est_rel_poses1 = torch.cat((gt_rel_poses[:, :3], q), dim=1)
                    ref_pose = utils.compute_abs_pose_torch(est_rel_poses1, minibatch.get('query_pose').float())
                    reproj_img = utils.reproject_RGB(minibatch.get('query_orig'), minibatch.get('depth'), minibatch.get('query_pose'), ref_pose)
                    reproj_img = reproj_img.clamp(0, 1)
                    cnt_non_zero = non_zero_mask.sum()
                    criterion1 = reproj_l2_loss(reproj_img * non_zero_mask, gt_reproj_orig)/cnt_non_zero
                    #if criterion1 > 0.01:
                    #    print(criterion1)
                    #criterion1 = reproj_l2_loss(reproj_img, gt_reproj_orig)
                    criterion += args.reproj_loss_w * criterion1
                    if i%100 == 0:
                        #gt_reproj_orig_ = reproj_transforms_inv(gt_reproj_orig)
                        #reproj_img_ = reproj_transforms_inv(reproj_img)
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
                            #rot_repr = utils.compute_rotation_matrix_from_ortho6d(rot_repr)
                            rot_repr = rotation_6d_to_matrix(rot_repr)
                        elif rot_repr_type == '9d':
                            rot_repr = est_rel_poses[:, 3:].reshape(batch_size, 3, 3).transpose(1,2)
                        #q = utils.compute_quaternions_from_rotation_matrices(rot_repr)
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

                    utils.log_to_tensorboard(writer, criterion.item(), criterion1.item(), i)

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

    elif args.mode == 'test': # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        if args.is_knn:
            test_dataset = KNNCameraPoseDataset(args.dataset_path, args.test_labels_file, args.refs_file, args.test_knn_file, transform, args.knn_len, args.knn_len, False)
        else:
            test_dataset = RelPoseDataset(args.dataset_path, args.test_labels_file, False, transform)

        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)
        pose_stats = np.zeros((len(dataloader.dataset), 3))
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
                        est_rel_pose = convert_to_quat(rot_repr_type, est_rel_pose)
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
                                                                            gt_rel_pose)
                            posit_err += curr_posit_err
                            orient_err += curr_orient_err

                        posit_err /= len(eval_reductions)
                        orient_err /= len(eval_reductions)

                    else:
                        est_rel_pose = convert_to_quat(rot_repr_type, est_rel_pose)
                        # Evaluate error
                        posit_err, orient_err = utils.pose_err(est_rel_pose, gt_rel_pose)
                
                # Collect statistics
                pose_stats[i, 0] = posit_err.item()
                pose_stats[i, 1] = orient_err.item()
                pose_stats[i, 2] = (tn - t0)*1000

                msg = "Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    pose_stats[i, 0],  pose_stats[i, 1],  pose_stats[i, 2])

                #posit_err, orient_err = utils.pose_err(minibatch['ref_pose'].to(device).to(dtype=torch.float32).detach(),
                #                                       minibatch['query_pose'].to(dtype=torch.float32).detach())
                #msg = msg + ", distance from neighbor images: {:.2f}[m], {:.2f}[deg]".format(posit_err.mean().item(),
                #                                                                             orient_err.mean().item())
                logging.info(msg)

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}, {:.3f}".format(np.nanmedian(pose_stats[:, 0]), np.nanmedian(pose_stats[:, 1])))
        logging.info("Mean RPR inference time:{:.2f}[ms]".format(np.mean(pose_stats)))

    else: # feature extraction

        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        if args.is_knn:
            test_dataset = KNNCameraPoseDataset(args.dataset_path, args.test_labels_file, args.refs_file,
                                                args.test_knn_file, transform, args.knn_len, args.knn_len, False)
        else:
            test_dataset = RelPoseDataset(args.dataset_path, args.test_labels_file, False, transform)

        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)

        out_file = 'my_knn.csv'
        knn_df = pd.read_csv(args.test_knn_file, header=None)
        f = open(out_file, "w")
        start_index = 0
        bExtractFeatures = True
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                query = minibatch.get('query')
                refs = minibatch.get('knn')

                if bExtractFeatures:
                    features_query = model.forward_backbone(query)
                    features_query = features_query.reshape(1, -1)
                    n, k, h, w, c = refs.shape
                    refs = refs.reshape(n * k, h, w, c)
                    features_refs = model.forward_backbone(refs)
                    features_refs = features_refs.reshape(n*k, -1)
                    indices = get_knn_indices(features_query, features_refs)
                    indices = indices.cpu().numpy()
                    print(indices)
                    f.write(knn_df.iloc[i][0] + ",")
                    for j in indices[start_index:(start_index + args.knn_len - 1)]:
                        f.write(knn_df.iloc[i][j+1] + ",")
                    f.write(knn_df.iloc[i][indices[start_index + args.knn_len - 1]+1] + "\n")

        f.close()









