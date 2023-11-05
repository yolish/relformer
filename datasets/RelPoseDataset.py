import random
from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import transforms3d as t3d
import os
import torchvision
from PIL import Image
from torchvision import transforms
import torch
class RelPoseDataset(Dataset):
    def __init__(self, data_path, pairs_file, is_reproj, transform=None, reproj_transform=None):
        self.img_path1, self.scenes1, self.scene_ids1, self.poses1, \
        self.img_path2, self.scenes2, self.scene_ids2, self.poses2, self.rel_poses = \
            read_pairs_file(data_path, pairs_file)
        self.transform = transform
        self.reproj_transform = reproj_transform
        self.data_path = data_path
        self.reproj_dir = '/reproj_pytorch/'
        self.is_reproj = is_reproj

    def __len__(self):
        return len(self.img_path1)

    def __getitem__(self, idx):
        img1 = imread(self.img_path1[idx])
        img2 = imread(self.img_path2[idx])
        pose1 = self.poses1[idx]
        pose2 = self.poses2[idx]
        rel_pose = self.rel_poses[idx]

        filename1 = os.path.splitext(os.path.basename(self.img_path1[idx]))[0]
        filename2 = os.path.splitext(os.path.basename(self.img_path2[idx]))[0]
        reproj_filename = self.data_path + self.scenes1[idx] + self.reproj_dir + filename1 + '_' + filename2 + '.png'

        orig_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

        if self.transform:
            img1_orig = orig_transform(img1)
            img2_orig = orig_transform(img2)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        is_flip = False
        # randomly flip images in an image pair
        if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            img1_orig, img2_orig = img2_orig, img1_orig
            pose1, pose2 = pose2, pose1
            rel_pose[:3] = -rel_pose[:3]
            rel_pose[3:] = [rel_pose[3], -rel_pose[4], -rel_pose[5], -rel_pose[6]]
            reproj_filename = self.data_path + self.scenes1[idx] + self.reproj_dir + filename2 + '_' + filename1 + '.png'
            is_flip = True

        img_reproj = None
        img_depth = None
        if self.is_reproj:
            img_reproj = imread(reproj_filename)
            if is_flip:
                img_depth = imread(self.img_path2[idx].replace('color.png', 'depth.png'))
            else:
                img_depth = imread(self.img_path1[idx].replace('color.png', 'depth.png'))

            img_depth = torch.from_numpy(img_depth.astype(np.float32)).unsqueeze(0)
            #img_reproj = Image.open(reproj_filename).convert("L")
            if self.reproj_transform:
                img_reproj_orig = orig_transform(img_reproj)
                img_reproj = self.reproj_transform(img_reproj)
                #print(reproj_filename)


            return {'query': img1,
                    'ref': img2,
                    'query_pose': pose1,
                    'ref_pose': pose2,
                    'rel_pose': rel_pose,
                    'reproj': img_reproj,
                    'depth': img_depth,
                    'query_orig': img1_orig,
                    'ref_orig': img2_orig,
                    'reproj_orig': img_reproj_orig}

        #todo add positive and negative

        return {'query': img1,
                'ref': img2,
                'query_pose': pose1,
                'ref_pose': pose2,
                'rel_pose':rel_pose}

def read_pairs_file(dataset_path, labels_file):
    df = pd.read_csv(labels_file)
    img_paths = []
    scenes = []
    scene_ids = []
    all_poses = []
    n = df.shape[0]
    for suffix in ["a", "b"]:
        img_paths.append([join(dataset_path, path) for path in df['img_path_{}'.format(suffix)].values])
        if "scene_{}".format(suffix) in df.keys():
            scenes.append(df['scene_{}'.format(suffix)].values)
            #scene_ids.append( df['scene_id_{}'.format(suffix)].values)
            scene_ids.append([])
        else:
            scenes.append([])
            scene_ids.append([])
        poses = np.zeros((n, 7))
        position_key = "x"
        if "x1_a" not in df.keys():
            position_key = "t"
        if "x1_a" not in df.keys() and "t1_a" not in df.keys():
            poses = np.zeros((n, 7))
        else:
            poses[:, 0] = df['{}1_{}'.format(position_key, suffix)].values
            poses[:, 1] = df['{}2_{}'.format(position_key, suffix)].values
            poses[:, 2] = df['{}3_{}'.format(position_key, suffix)].values
            poses[:, 3] = df['q1_{}'.format(suffix)].values
            poses[:, 4] = df['q2_{}'.format(suffix)].values
            poses[:, 5] = df['q3_{}'.format(suffix)].values
            poses[:, 6] = df['q4_{}'.format(suffix)].values
        all_poses.append(poses)
    #img_paths1, img_paths2 = img_paths
    scenes1, scenes2 = scenes
    scene_ids1, scene_ids2 = scene_ids
    poses1, poses2 = all_poses
    rel_poses = np.zeros((n, 7))

    img_paths1 = []
    img_paths2 = []
    for i in range(len(df['img_path_a'])):
        if scenes1[i] in df['img_path_a'][i]:
            img_paths1.append(join(dataset_path, df['img_path_a'][i]))
            img_paths2.append(join(dataset_path, df['img_path_b'][i]))
        else:
            img_paths1.append(join(dataset_path, scenes1[i] + df['img_path_a'][i]))
            img_paths2.append(join(dataset_path, scenes2[i] + df['img_path_b'][i]))

    suffix = "ab"
    if "x1_ab" in df.keys():
        rel_poses[:, 0] = df['x1_{}'.format(suffix)].values
        rel_poses[:, 1] = df['x2_{}'.format(suffix)].values
        rel_poses[:, 2] = df['x3_{}'.format(suffix)].values
        rel_poses[:, 3] = df['q1_{}'.format(suffix)].values
        rel_poses[:, 4] = df['q2_{}'.format(suffix)].values
        rel_poses[:, 5] = df['q3_{}'.format(suffix)].values
        rel_poses[:, 6] = df['q4_{}'.format(suffix)].values
    else:
        for i, p1 in enumerate(poses1):
            p2 = poses2[i]
            x_rel, q_rel =  compute_rel_pose(p1, p2)
            rel_poses[i, :3]  = x_rel
            rel_poses[i, 3:] = q_rel


    return img_paths1, scenes1, scene_ids1, poses1, img_paths2, scenes2, scene_ids2, poses2, rel_poses

def compute_rel_pose(p1, p2):
    t1 = p1[:3]
    q1 = p1[3:]
    rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

    t2 = p2[:3]
    q2 = p2[3:]
    rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

    t_rel = t2 - t1
    rot_rel = np.dot(np.linalg.inv(rot1), rot2)
    q_rel = t3d.quaternions.mat2quat(rot_rel)
    return t_rel, q_rel