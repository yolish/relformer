from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
from skimage.io import imread
import torch
import transforms3d as t3d
from torchvision import transforms
import torch
from pathlib import Path
import os
import cv2
class KNNCameraPoseDataset(Dataset):
    """
        A class representing a dataset of query and its knns
    """

    def __init__(self, dataset_path, query_labels_file, db_labels_file, knn_file,
                 transform, knn_len, sample_size=1, sample=True, is_reproj=False):
        super(KNNCameraPoseDataset, self).__init__()

        self.query_img_paths, self.query_poses, self.query_scenes, self.query_scenes_ids = read_labels_file(query_labels_file, dataset_path)
        self.query_to_pose = dict(zip(self.query_img_paths, self.query_poses))

        self.db_img_paths, self.db_poses, self.db_scenes, self.db_scenes_ids = read_labels_file(db_labels_file, dataset_path)
        self.db_to_pose = dict(zip(self.db_img_paths, self.db_poses))

        knns = {}
        lines = open(knn_file).readlines()
        i = 0
        knn_queries = {}
        depth_queries = {}
        for l in lines:
            neighbors = l.rstrip().split(",")
            nn = neighbors[0]
            if 'netvlad.npz' in nn:
                nn = neighbors[0].replace('_netvlad.npz', '.png')
                nn = nn.replace('_', '/')
            if nn[0] == '/':
                nn = nn[1:]
            q = join(dataset_path, nn)

            if is_reproj:
                filename1 = os.path.splitext(os.path.basename(q))[0]
                path1 = Path(q)
                dir1 = os.path.basename(path1.parent)
                scene1 = os.path.basename(path1.parent.parent)
                depth_file_name = dataset_path + scene1 + '_depth/' + dir1 + '_' + filename1 + '.depth.tiff'
                if not os.path.exists(depth_file_name):
                    depth_file_name = dataset_path + scene1 + '_depth/' + dir1 + '_' + filename1 + '.depth.png'
                    if not os.path.exists(depth_file_name):
                        print("missing depth file: " + depth_file_name)
                        continue
            my_knns = []
            for nn in neighbors[1:]:
                if 'netvlad.npz' in nn:
                    nn = nn.replace('_netvlad.npz', '.png')
                    nn = nn.replace('_', '/')
                if nn is not "" and nn[0] == '/':
                    nn = nn[1:]
                if nn != '':
                    my_knns.append(join(dataset_path, nn))
            if len(my_knns)<1:
                continue
            if len(my_knns) < sample_size:
                num_to_add = sample_size-len(my_knns)
                for j in range(num_to_add):
                    my_knns.append(my_knns[j])
            knns[q] = my_knns
            knn_queries[i] = q
            i += 1
        self.knns = knns
        self.knn_queries = knn_queries
        self.sample_size = sample_size
        self.transform = transform
        self.sample = sample
        self.knn_len = knn_len
        self.dataset_path = dataset_path
        self.reproj_dir = '/reproj/'
        self.is_reproj = is_reproj
        self.orig_transform = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((270, 480)),
                                                  transforms.ToTensor(),
                                                  ])

    def __len__(self):
        return len(self.knn_queries)

    def load_img(self, img_path):
        img = imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img

    def load_orig_img(self, img_path):
        img = imread(img_path)
        if self.transform:
            img = self.orig_transform(img)
        return img

    def __getitem__(self, idx):
        query_path = self.knn_queries[idx]
        knn_paths = self.knns[query_path]

        if self.sample:
            indices = np.random.choice(min(len(knn_paths), self.knn_len), size=self.sample_size)
            knn_paths = np.array(knn_paths)[indices]
        else:
            knn_paths = knn_paths[:self.sample_size]

        query = self.load_img(query_path)
        query_pose = self.query_to_pose[query_path]

        filename1 = os.path.splitext(os.path.basename(query_path))[0]
        path1 = Path(query_path)
        dir1 = os.path.basename(path1.parent)
        scene1 = os.path.basename(path1.parent.parent)

        query_orig = 0
        knn_orig = 0
        img_reproj = 0
        img_depth = 0
        if self.is_reproj:
            query_orig = self.load_orig_img(query_path)
            depth_file_name = self.dataset_path + scene1 + '_depth/' + dir1 + '_' + filename1 + '.depth.tiff'
            if not os.path.exists(depth_file_name):
                depth_file_name = self.dataset_path + scene1 + '_depth/' + dir1 + '_' + filename1 + '.depth.png'
                if not os.path.exists(depth_file_name):
                    print("missing depth file: " + depth_file_name)
                    return None
            img_depth = cv2.imread(depth_file_name, -1)
            img_depth = cv2.resize(img_depth, (480, 270))
            # img_depth = imread(depth_file_name)
            img_depth = torch.from_numpy(img_depth.astype(np.float32)).unsqueeze(0)

        if self.sample_size == 1:
            ref = self.load_img(knn_paths[0])
            ref_pose = self.db_to_pose[knn_paths[0]]

            rel_pose = np.zeros(7)
            x_rel, q_rel = compute_rel_pose(query_pose, ref_pose)
            rel_pose[:3] = x_rel
            rel_pose[3:] = q_rel

            if self.is_reproj:
                knn_orig = self.load_orig_img(knn_paths[0])
                filename2 = os.path.splitext(os.path.basename(knn_paths[0]))[0]
                path2 = Path(knn_paths[0])
                dir2 = os.path.basename(path2.parent)
                reproj_filename = self.dataset_path + scene1 + self.reproj_dir + dir1 + '_' + filename1 + '_' + dir2 + '_' + filename2 + '.png'
                img_reproj = self.load_orig_img(reproj_filename)

        else:
            knn = []
            knn_imgs = []
            knn_poses = np.zeros((self.sample_size, 7))
            #knn_poses = np.zeros((len(knn_paths), 7))
            for i, nn_path in enumerate(knn_paths):
                if 'png' not in nn_path:
                    continue
                knn.append(self.load_img(nn_path))
                knn_poses[i, :] = self.db_to_pose[nn_path]
                knn_imgs.append(nn_path)

            ref = torch.stack(knn)
            ref_pose = knn_poses
        
            rel_pose = np.zeros((self.sample_size, 7))
            for i in range(self.sample_size):
                x_rel, q_rel = compute_rel_pose(query_pose,  knn_poses[i])                            
                rel_pose[i, :] = np.concatenate([x_rel, q_rel])
            
        return {'query': query,
                'ref': ref,
                'query_pose': query_pose,
                'ref_pose': ref_pose,
                'rel_pose': rel_pose,
                'depth': img_depth,
                'query_orig': query_orig,
                'ref_orig': knn_orig,
                'reproj_orig': img_reproj
                }
    

def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids

def read_pairs_file_(dataset_path, labels_file):
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
            scene_ids.append( df['scene_id_{}'.format(suffix)].values)
        else:
            scenes.append([])
            scene_ids.append([])
        poses = np.zeros((n, 7))
        position_key = "x"
        if "x1_a" not in df.keys():
            position_key = "t"
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

    for i, p1 in enumerate(poses1):
        p2 = poses2[i]
        x_rel, q_rel = compute_rel_pose(p1, p2)
        rel_poses[i, :3]  = x_rel
        rel_poses[i, 3:] = q_rel

    return imgs_paths, poses, scenes, scenes_ids


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


