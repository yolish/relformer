import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import mkdir, getcwd
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import ssl
import os
import sys
import torchvision
import torchgeometry as tgm
from pytorch3d.transforms import quaternion_to_matrix
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import datetime
import transforms3d as t3d

def set_proxy():
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["http_proxy"] = "http://127.0.0.1:3128"
    os.environ["https_proxy"] = "http://127.0.0.1:3128"
    os.environ["ftp_proxy"] = "http://127.0.0.1:3128"
    os.environ["socks_proxy"] = "http://127.0.0.1:3128"

# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def create_output_dir(name):
    """
    Create a new directory for outputs, if it does not already exist
    :param name: (str) the name of the directory
    :return: the path to the outpur directory
    """
    out_dir = join(getcwd(), name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger():
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M_%S", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = create_output_dir('out')

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)



# Evaluation utils
##########################
def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.clamp(torch.abs(inner_prod), -1, 1)) * 180 / np.pi
    #orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err

def compose(t, rot):
    B = t.size(0)
    x = torch.eye(4)
    x = x.reshape((1, 4, 4))
    x = x.repeat(B, 1, 1)
    x[:,:3,:3] = rot
    x[:,:3,3] = t
    return x

def reproject_RGB(rgb_img, depth_img, pose1, pose2):
    K = np.mat([[585, 0, 320, 0], [0, 585, 240, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t1 = pose1[:,:3]
    t2 = pose2[:,:3]
    q1 = pose1[:,3:]
    q2 = pose2[:,3:]
    #rot1 = t3d.quaternions.quat2mat(q1 / torch.linalg.norm(q1))
    rot1 = quaternion_to_matrix(q1 / torch.linalg.norm(q1))
    #pose_mat1 = t3d.affines.compose(t1, rot1, torch.ones(3))  # 4x4
    pose_mat1 = compose(t1, rot1)
    extrinsics1 = torch.linalg.inv(pose_mat1).to(rgb_img.device)
    #extrinsics1 = torch.transpose(pose_mat1,1,2).to(rgb_img.device)
    #rot2 = t3d.quaternions.quat2mat(q2 / torch.linalg.norm(q2))
    rot2 = quaternion_to_matrix(q2 / torch.linalg.norm(q2))
    #pose_mat2 = t3d.affines.compose(t2, rot2, torch.ones(3))  # 4x4
    pose_mat2 = compose(t2, rot2)
    extrinsics2 = torch.linalg.inv(pose_mat2).to(rgb_img.device)
    #extrinsics2 = torch.transpose(pose_mat2,1,2).to(rgb_img.device)

    reproj_img = reproject_RGB_kornia(rgb_img, depth_img, K, K, extrinsics1, extrinsics2)

    return reproj_img

def reproject_RGB_kornia(rgb_img, depth_img, K_ref, K_cam, Rt_ref, Rt_cam, width=640, height=480):
    B = rgb_img.size(0)
    height = torch.tensor([int(height)]).to(rgb_img.device)
    width = torch.tensor([int(width)]).to(rgb_img.device)
    K_ref = torch.from_numpy(K_ref).unsqueeze(0).float().to(rgb_img.device)
    K_ref = K_ref.repeat(B, 1, 1)
    K_cam = torch.from_numpy(K_cam).unsqueeze(0).float().to(rgb_img.device)
    K_cam = K_cam.repeat(B, 1, 1)
     # pinholes camera models
    pinhole_dst = tgm.PinholeCamera(intrinsics=K_ref, extrinsics=Rt_ref, height=height, width=width)
    pinhole_src = tgm.PinholeCamera(intrinsics=K_cam, extrinsics=Rt_cam, height=height, width=width)
    # warp the destionation frame to reference by depth
    image_src = tgm.depth_warp(pinhole_dst, pinhole_src,
        depth_img, rgb_img.float(), height[0], width[0])  # NxCxHxW
    return image_src


# Augmentations
train_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

}
test_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        ])
}

reproj_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.5], [0.5])
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        ])
}

reproj_transforms_inv = {
    'baseline': transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
}

reproj_transforms_depth = {
    'baseline': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()
        ])
}

train_transforms_vae = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

}

# Code from: https://github.com/JYChen18/RPMG/
# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# quaternion batch*4
def compute_rotation_matrix_from_quaternion(quaternion, n_flag=True):
    batch = quaternion.shape[0]
    if n_flag:
        quat = normalize_vector(quaternion)
    else:
        quat = quaternion
    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# quaternions batch*4,
# matrices batch*4*4 or batch*3*3
def compute_quaternions_from_rotation_matrices(matrices):
    batch = matrices.shape[0]

    w = torch.sqrt(
        torch.max(1.0 + matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2], torch.zeros(1).to(matrices.device))) / 2.0
    w = torch.max(w, torch.autograd.Variable(torch.zeros(batch).to(matrices.device)) + 1e-8)  # batch
    w4 = 4.0 * w
    x = (matrices[:, 2, 1] - matrices[:, 1, 2]) / w4
    y = (matrices[:, 0, 2] - matrices[:, 2, 0]) / w4
    z = (matrices[:, 1, 0] - matrices[:, 0, 1]) / w4
    quats = torch.cat((w.view(batch, 1), x.view(batch, 1), y.view(batch, 1), z.view(batch, 1)), 1)
    quats = normalize_vector(quats)
    return quats


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

# Code from https://github.com/amakadia/svd_for_pose
def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r

def compute_abs_pose(rel_pose, abs_pose_neighbor, device):
    # p_neighbor p_rel = p_query
    # p1 p_rel = p2
    abs_pose_query = torch.zeros_like(rel_pose)
    rel_pose = rel_pose.cpu().numpy()
    abs_pose_neighbor = abs_pose_neighbor.cpu().numpy()
    for i, rpr in enumerate(rel_pose):
        p1 = abs_pose_neighbor[i]

        t_rel = rpr[:3]
        q_rel = rpr[3:]
        rot_rel = t3d.quaternions.quat2mat(q_rel/ np.linalg.norm(q_rel))

        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1/ np.linalg.norm(q1))

        t2 = t1 + t_rel
        rot2 = np.dot(rot1,rot_rel)
        q2 = t3d.quaternions.mat2quat(rot2)
        abs_pose_query[i][:3] = torch.Tensor(t2).to(device)
        abs_pose_query[i][3:] = torch.Tensor(q2).to(device)

    return abs_pose_query
def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)
    return q

def compute_abs_pose_torch(rel_pose, abs_pose_neighbor):
    abs_pose_query = torch.zeros_like(rel_pose)
    abs_pose_query[:, :3] = abs_pose_neighbor[:, :3] + rel_pose[:, :3]
    abs_pose_query[:, 3:] = qmult(abs_pose_neighbor[:, 3:], rel_pose[:, 3:])
    return abs_pose_query

def init_tensorbaord_log(args, saved_path='default', save_dir=None):
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    if save_dir is None:
        save_dir = os.path.join("{}/tf_{}".format(saved_path, current_time))
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    # Saving run line
    with open(os.path.join(save_dir, 'run_line.txt'), 'w') as f:
        f.write("python {}".format(' '.join(sys.argv[:])))
        print("python {}".format(' '.join(sys.argv[:])))
    # Saving arguments to json
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print("Saving arguments to: {}".format(os.path.join(save_dir, 'args.txt')))
    return writer, save_dir

def log_to_tensorboard(writer, loss1, loss2, step):
    with torch.no_grad():
        writer.add_scalar('train/total_loss', loss1, step)
        writer.add_scalar('train/reproj_loss', loss2, step)

def log_img_to_tensorboard( writer, gt_reproj_orig, reproj_img, step):
    with torch.no_grad():
        batch_size = gt_reproj_orig.shape[0]
        images_0_grid = make_grid(gt_reproj_orig[:min(3, batch_size)], nrow=min(3, batch_size), normalize=False)
        images_1_grid = make_grid(reproj_img[:min(3, batch_size)], nrow=min(3, batch_size), normalize=False)
        image_grid = torch.cat((images_0_grid, images_1_grid), 1)
        writer.add_image('train/reproj', image_grid, step)

def log_img_to_tensorboard_triplet( writer, gt_reproj_orig, reproj_img, step):
    with torch.no_grad():
        batch_size = gt_reproj_orig.shape[0]
        images_0_grid = make_grid(gt_reproj_orig[:min(3, batch_size)], nrow=min(3, batch_size), normalize=False)
        images_1_grid = make_grid(reproj_img[:min(3, batch_size)], nrow=min(3, batch_size), normalize=False)
        image_grid = torch.cat((images_0_grid, images_1_grid), 2)

        writer.add_image('train/reproj', image_grid, step)