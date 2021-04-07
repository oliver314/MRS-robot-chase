#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os


# Map occupancy information
size_m = 10         # 10m x 10m size
resolution_m = 0.01 # 1cm resolution
size_px = int(size_m/resolution_m) # image is of size: size_px x size_px

# Limits (in meters) for x and y in the environment
xlim = [-4, 4]
ylim = [-4, 4]

# load world image
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.png")
map_img = cv2.imread(path, 0)


robot_point = np.array([ 1.49956512 , 0.50019956])
pc1 = np.array([-0.98360127 , 0.49998751])
pc_list = [pc1]

def world_to_pixel(point):
    xy = np.true_divide(point, resolution_m)
    xy[1] *= -1
    xy += size_px / 2
    xy = xy.astype(np.int32)
    return xy

def obstructed(point1, point2):
    p1_px = world_to_pixel(point1)
    p2_px = world_to_pixel(point2)
    obstructed = False

    print(point1, point2)
    print(p1_px, p2_px)
    if abs(p1_px[0] - p2_px[0]) > abs(p1_px[1] - p2_px[1]):
        if p1_px[0] > p2_px[0]:
            start_pt = p2_px
            end_pt = p1_px
        else:
            start_pt = p1_px
            end_pt = p2_px

        grad = (end_pt[1] - start_pt[1]) / (end_pt[0] - start_pt[0])

        y = start_pt[1]
        x = start_pt[0]
        for i in range(end_pt[0] - start_pt[0]):
            x_px = int(x + i)
            y_px = int(y)
            if map_img[y_px][x_px] == 0:
                obstructed = True
                break
            y += grad

    else:
        if p1_px[1] > p2_px[1]:
            start_pt = p2_px
            end_pt = p1_px
        else:
            start_pt = p1_px
            end_pt = p2_px

        grad = (end_pt[0] - start_pt[0]) / (end_pt[1] - start_pt[1])

        y = start_pt[1]
        x = start_pt[0]
        for i in range(end_pt[1] - start_pt[1]):
            x_px = int(x)
            y_px = int(y + i)
            if map_img[y_px][x_px] == 0:
                obstructed = True
                break
            x += grad
    return obstructed


dist_cutoff = 3
undetected_set = []
detectable_set = []

# First check distance from robot
for pc in pc_list:
    if np.linalg.norm(robot_point - pc) > 3:
        undetected_set.append(pc)
    else:
        detectable_set.append(pc)

viewable_set = []
# Now check if detectable set crosses any obstacles
for pc in detectable_set:
    if obstructed(pc,robot_point):
        undetected_set.append(pc)
    else:
        viewable_set.append(pc)

print(viewable_set, undetected_set)
#if not np.any((coord < 0) | (coord > size_px)) and map_img[coord[1]][coord[0]] == 255:
#    robot_PC.append(xy_raw[i])
