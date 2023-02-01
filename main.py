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
from models.pose_losses import CameraPoseLoss
from os.path import join
from models.relformer.RelFormer import RelFormer, RelFormer2, BrRwlFormer
from models.DeltaNet import DeltaNet

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("rpr_backbone_path", help="path to the backbone path")
    arg_parser.add_argument("labels_file", help="pairs file")
    arg_parser.add_argument("config_file", help="path to configuration file")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained PAE-based RPR model")

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

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.fdeterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # relformer
    arch = config.get("arch")
    if config.get("relformer2"):
        model = RelFormer2(config, args.rpr_backbone_path).to(device)
    elif arch == "relformer":
        model = RelFormer(config, args.rpr_backbone_path).to(device)
    elif arch == "b-relformer":
        model = BrRwlFormer(config, args.rpr_backbone_path).to(device)
    elif arch == "deltanet":
        model = DeltaNet(config, args.rpr_backbone_path).to(device)
    else:
        raise NotImplementedError(arch)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        train_w_triplet_loss = False # read from config
        #TODO Triplet loss

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
        train_dataset = RelPoseDataset(args.dataset_path, args.labels_file, transform)

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

        for epoch in range(n_epochs):
            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_rel_poses = minibatch['rel_pose'].to(dtype=torch.float32)
                batch_size = gt_rel_poses.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                neighbor_poses = minibatch['ref_pose'].to(device).to(dtype=torch.float32)
                 # Estimate the relative pose
                # Zero the gradients
                optim.zero_grad()
                res = model(minibatch)

                # TODO triplet loss
                if train_w_triplet_loss:
                    z_query = res.get("z_query")
                    z_neg = res.get("z_pos")
                    z_pos = res.get("z_pos")
                    #TODO compute triplet_loss

                est_rel_poses = res.get("rel_pose")

                criterion = pose_loss(est_rel_poses, gt_rel_poses)
                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_rel_poses.detach(), gt_rel_poses.detach())
                    msg = "[Batch-{}/Epoch-{}] running relative camera pose loss: {:.3f}, camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_freq_print),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item())
                    posit_err, orient_err = utils.pose_err(neighbor_poses.detach(), minibatch['query_pose'].to(dtype=torch.float32).detach())
                    msg = msg + ", distance from neighbor images: {:.2f}[m], {:.2f}[deg]".format(posit_err.mean().item(),
                                                                        orient_err.mean().item())
                    logging.info(msg)
                    # Resetting temporal loss used for logging
                    running_loss = 0.0
                    n_samples = 0

            # Save checkpoint3n
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_relformer_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_relformer_final.pth')


    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        test_dataset = RelPoseDataset(args.dataset_path, args.labels_file, transform)
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
                t0 = time.time()
                res = model(minibatch)
                est_rel_pose = res['rel_pose']
                torch.cuda.synchronize()
                tn = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_rel_pose, gt_rel_pose)

                # Collect statistics
                pose_stats[i, 0] = posit_err.item()
                pose_stats[i, 1] = orient_err.item()
                pose_stats[i, 2] = (tn - t0)*1000


                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    pose_stats[i, 0],  pose_stats[i, 1],  pose_stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(pose_stats[:, 0]), np.nanmedian(pose_stats[:, 1])))
        logging.info("Mean RPR inference time:{:.2f}[ms]".format(np.mean(pose_stats)))






