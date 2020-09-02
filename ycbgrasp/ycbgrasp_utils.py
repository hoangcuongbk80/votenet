# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Provides Python helper function to read My YCB object grasp dataset.

Author: Dinh-Cuong Hoang
Date: August, 2020

'''

import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points
import pc_util
import math

""" type2class={'007_tuna_fish_can':0, '008_pudding_box':1, '011_banana':2, '024_bowl':3, '025_mug':4, '044_flat_screwdriver':5,
            '051_large_clamp':6, '055_baseball':7, '061_foam_brick': 8, '065-h_cups':9} """

type2class={'011_banana':0, '024_bowl':1, '025_mug':2, '044_flat_screwdriver':3,'051_large_clamp':4}
class2type = {type2class[t]:t for t in type2class}

class YCBObject(object):
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.centroid = np.array([data[1],data[2],data[3]])
        self.w = data[4]
        self.l = data[5]
        self.h = data[6]
        self.heading_angle = data[7]
        if self.classname=='011_banana':
            self.heading_angle = data[7] - math.pi/8
        if self.classname=='044_flat_screwdriver':
            self.heading_angle = data[7] + math.pi/4
        if self.classname=='051_large_clamp':
            self.heading_angle = data[7]


def load_pointcloud(pc_filename):
    pointcloud = pc_util.read_ply(pc_filename)
    return pointcloud

def load_obb(obb_filename):
    lines = [line.rstrip() for line in open(obb_filename)]
    objects = [YCBObject(line) for line in lines[1:]]
    return objects

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds
