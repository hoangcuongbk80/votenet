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

type2class={'007_tuna_fish_can':0, '008_pudding_box':1, '011_banana':2, '024_bowl':3, '025_mug':4, '044_flat_screwdriver':5,
            '051_large_clamp':6, '055_baseball':7, '061_foam_brick': 8, '065-h_cups':9}
class2type = {type2class[t]:t for t in type2class}

class YCBObject(object):
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.centroid = np.array([data[1],data[2],data[3]])
        self.w = data[6]
        self.l = data[5]
        self.h = data[4]
        self.heading_angle = math.pi/2 + data[7]

def load_pointcloud(pc_filename):
    pointcloud = pc_util.read_ply(pc_filename)
    print(pointcloud)
    return pointcloud

def load_obb(obb_filename):
    lines = [line.rstrip() for line in open(obb_filename)]
    objects = [YCBObject(line) for line in lines[1:]]
    return objects
