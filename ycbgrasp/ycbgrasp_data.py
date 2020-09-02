# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: December, 2018

Note: removed unused code for frustum preparation.
Changed a way for data visualization (removed depdency on mayavi).
Load depth with scipy.io
'''

import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
import pc_util
import ycbgrasp_utils

parser = argparse.ArgumentParser()
parser.add_argument('--viz', action='store_true', help='Run data visualization.')
parser.add_argument('--gen_data', action='store_true', help='Generate training dataset.')
parser.add_argument('--num_sample', type=int, default=1000, help='Number of samples [default: 10000]')

args = parser.parse_args()

DEFAULT_TYPE_WHITELIST = ['011_banana','024_bowl','025_mug','044_flat_screwdriver', '051_large_clamp']

class ycb_object(object):
    ''' Load and parse object data '''
    def __init__(self, data_dir):

        self.pointcloud_dir = os.path.join(data_dir, 'pointcloud')
        self.obb_dir = os.path.join(data_dir, 'obb')
        self.num_samples = args.num_sample
        
    def __len__(self):
        return self.num_samples

    def get_pointcloud(self, idx):
        pc_filename = os.path.join(self.pointcloud_dir, '%d.ply'%(idx))
        print(pc_filename)
        return ycbgrasp_utils.load_pointcloud(pc_filename)

    def get_obb(self, idx): 
        obb_filename = os.path.join(self.obb_dir, '%d.txt'%(idx))
        print(obb_filename)
        return ycbgrasp_utils.load_obb(obb_filename)

def data_viz(data_dir, dump_dir=os.path.join(BASE_DIR, 'data_viz_dump')):
    ''' Examine and visualize ycbgrasp dataset. '''
    ycb = ycb_object(data_dir)
    idxs = np.array(range(0,len(ycb)))
    np.random.seed(0)
    np.random.shuffle(idxs)

    if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

    for idx in range(len(ycb)):
        if idx%30:
            continue
        data_idx = idxs[idx]
        print('data index: ', data_idx)
        pc = ycb.get_pointcloud(data_idx)
        print('Point cloud shape:', pc.shape)
        objects = ycb.get_obb(data_idx)
        oriented_boxes = []
        for obj in objects:
            obb = np.zeros((7))
            obb[0:3] = obj.centroid
            obb[3:6] = np.array([obj.l,obj.w,obj.h])*2
            obb[6] = obj.heading_angle
            print('Object cls, heading, l, w, h:',\
                 obj.classname, obj.heading_angle, obj.l, obj.w, obj.h)
            oriented_boxes.append(obb)
        if len(oriented_boxes)>0:
            oriented_boxes = np.vstack(tuple(oriented_boxes))
            pc_util.write_oriented_bbox(oriented_boxes,
            os.path.join(dump_dir, str(idx) + '_obbs.ply'))
            pc_util.write_ply(pc, os.path.join(dump_dir, str(idx) + '_pc.ply'))
        
    print('Complete!')
    
def extract_ycbgrasp_data(data_dir, idx_filename, output_folder, num_point=20000,
    type_whitelist=DEFAULT_TYPE_WHITELIST):
    
    dataset = ycb_object(data_dir)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_obb(data_idx)

        # Save obb and point cloud
        object_list = []
        for obj in objects:
            obb = np.zeros((8))
            obb[0:3] = obj.centroid
            obb[3:6] = np.array([obj.l,obj.w,obj.h])
            obb[6] = obj.heading_angle
            obb[7] = ycbgrasp_utils.type2class[obj.classname]
            object_list.append(obb)

        if len(object_list)==0:
            obbs = np.zeros((0,8))
        else:
            obbs = np.vstack(object_list) # (K,8)

        pc = dataset.get_pointcloud(data_idx)

        np.savez_compressed(os.path.join(output_folder,'%06d_pc.npz'%(data_idx)), pc=pc)
        np.save(os.path.join(output_folder, '%06d_bbox.npy'%(data_idx)), obbs)

        # Compute and save votes
        N = pc.shape[0]
        point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
        point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
        indices = np.arange(N)
        for obj in objects:
            # Find all points in this object's OBB
            box3d_pts_3d = ycbgrasp_utils.my_compute_box_3d(obj.centroid,
                np.array([obj.l,obj.w,obj.h]), obj.heading_angle)
            pc_in_box3d,inds = ycbgrasp_utils.extract_pc_in_box3d(pc, box3d_pts_3d)
            # Assign first dimension to indicate it is in an object box
            point_votes[inds,0] = 1
            # Add the votes (all 0 if the point is not in any object's OBB)
            votes = np.expand_dims(obj.centroid,0) - pc_in_box3d[:,0:3]
            sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
            for i in range(len(sparse_inds)):
                j = sparse_inds[i]
                point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
                # Populate votes with the fisrt vote
                if point_vote_idx[j] == 0:
                    point_votes[j,4:7] = votes[i,:]
                    point_votes[j,7:10] = votes[i,:]
            point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
        np.savez_compressed(os.path.join(output_folder, '%06d_votes.npz'%(data_idx)), point_votes = point_votes)
        print(point_votes)
    return 0

    
if __name__=='__main__':
    
    if args.viz:
        data_viz(os.path.join(BASE_DIR, 'data'))
        exit()

    if args.gen_data:
        idxs = np.array(range(0,args.num_sample))
        np.random.seed(0)
        np.random.shuffle(idxs)
        np.savetxt(os.path.join(BASE_DIR, 'data', 'train_data_idx.txt'), idxs[:800], fmt='%i')
        np.savetxt(os.path.join(BASE_DIR, 'data', 'val_data_idx.txt'), idxs[800:], fmt='%i')
        
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        extract_ycbgrasp_data(DATA_DIR, os.path.join(DATA_DIR, 'train_data_idx.txt'),
            output_folder = os.path.join(DATA_DIR, 'train'), num_point=50000)
        extract_ycbgrasp_data(DATA_DIR, os.path.join(DATA_DIR, 'val_data_idx.txt'),
            output_folder = os.path.join(DATA_DIR, 'val'), num_point=50000)