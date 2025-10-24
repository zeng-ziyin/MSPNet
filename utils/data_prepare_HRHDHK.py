from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

def read_from_txt(path_txt):
    file = open(path_txt, 'r')
    data = []
    for line in file:
        data.append(line.split('\n')[0])
    return data


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, required=True, help='the number of GPUs to use [default: 0]')
    # FLAGS = parser.parse_args()
    dataset_name = 'HRHDHK'
    dataset_path = '/media/zengziyin/ZZY/Data/HRHD-HK'
    preparation_types = ['grid']  # Grid sampling & Random sampling
    grid_size = 0.2
    random_sample_ratio = 10
    # train_files = np.sort([join(dataset_path, 'train', i) for i in os.listdir(join(dataset_path, 'train'))])
    # test_files = np.sort([join(dataset_path, 'test', i) for i in os.listdir(join(dataset_path, 'test'))])
    # val_files = np.sort([join(dataset_path, 'val', i) for i in os.listdir(join(dataset_path, 'val'))])
    train_files = read_from_txt('/media/zengziyin/ZZY/Data/HRHD-HK/trainfile.txt')
    val_files = read_from_txt('/media/zengziyin/ZZY/Data/HRHD-HK/valfile.txt')
    test_files = read_from_txt('/media/zengziyin/ZZY/Data/HRHD-HK/testfile.txt')
    files = np.sort([join(dataset_path, 'all', i) for i in os.listdir(join(dataset_path, 'all'))])

    for sample_type in preparation_types:
        for pc_path in files:
            cloud_name = pc_path.split('/')[-1][:-4]
            print('start to process:', cloud_name)

            # create output directory
            out_folder = join(dataset_path, sample_type + '_{:.3f}'.format(grid_size))
            os.makedirs(out_folder) if not exists(out_folder) else None
            print(out_folder)

            # check if it has already calculated
            if exists(join(out_folder, cloud_name + '_KDTree.pkl')):
                print(cloud_name, 'already exists, skipped')
                continue

            # if pc_path in train_files:
            xyz, rgb, labels = DP.read_ply_data(pc_path, with_rgb=True)
            labels -= 1
            # else:
            #     xyz, rgb = DP.read_ply_data(pc_path, with_rgb=True, with_label=False)
            #     labels = np.zeros(len(xyz), dtype=np.uint8)

            sub_ply_file = join(out_folder, cloud_name + '.ply')
            if sample_type == 'grid':
                sub_xyz, sub_rgb, sub_labels = DP.grid_sub_sampling(xyz, rgb, labels, grid_size)
            else:
                sub_xyz, sub_rgb, sub_labels = DP.random_sub_sampling(xyz, rgb, labels, random_sample_ratio)

            sub_rgb = sub_rgb / 255.0
            sub_labels = np.squeeze(sub_labels)
            write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            search_tree = KDTree(sub_xyz, leaf_size=50)
            kd_tree_file = join(out_folder, cloud_name + '_KDTree.pkl')
            with open(kd_tree_file, 'wb') as f:
                pickle.dump(search_tree, f)

            proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            proj_save = join(out_folder, cloud_name + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)

