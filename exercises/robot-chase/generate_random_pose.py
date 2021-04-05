import random
import numpy as np
import os
import cv2

# Map occupancy information
size_m = 10         # 10m x 10m size
resolution_m = 0.01 # 1cm resolution
size_px = int(size_m/resolution_m) # image is of size: size_px x size_px


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.png")
map = cv2.imread(path, 0)

def generate_random_pose(img, xlim, ylim):
  xrng = abs(xlim[1] - xlim[0])
  xmid = (xlim[1] + xlim[0])/2

  yrng = abs(ylim[1] - ylim[0])
  ymid = (ylim[1] + ylim[0]) / 2

  while 1:
    x = xrng * random.random() + xmid
    y = yrng * random.random() + ymid

    pose = np.array([(x/resolution_m) + size_px/2, (-y/resolution_m) + size_px/2])
    pose = pose.astype(np.int32)

    if not np.any((pose < 0) | (pose > size_px)) and img[pose[1]][pose[0]] == 255:
      break

  return "-x " + str(round(x,2)) + " -y " + str(round(y,2)) + " -z 0"

if __name__ == "__main__":
    xlim = [-4,4]
    ylim = [-4,4]
    pose = generate_random_pose(map, xlim, ylim)
    print(pose)