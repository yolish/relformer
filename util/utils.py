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