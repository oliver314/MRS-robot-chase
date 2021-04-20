from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os


size_m = 10         # 10m x 10m size
resolution_m = 0.01 # 1cm resolution
size_px = int(size_m/resolution_m) # image is of size: size_px x size_px

# set inflation of lines. (centre to edge inflation distance)
inflation_m = 0.1
draw_thickness = int(2 * inflation_m / resolution_m)


# Define the world here (assume rectangle)
# horizontal, vertical, centre x, centre y, angle=0, fill=-1
world = [8, 8, 0, 0, 0, -1]

# Define rectangles here
# horizontal, vertical, centre x, centre y, angle (degrees), fill
rect1 = [8.15, 8.15, 0, 0, 0, 0]

''' Cluttered world stuff: '''
rect2 = [0.2, 5, 2.12, -0.28, 0, 1]
rect3 = [1.65, 0.25, 1.41, 2.45, 0, 1]
rect4 = [1.3, 1.33, -1.94, 1.7, 0, 1]
rect5 = [1.33, 1.32, -1.91, -2.34, 0, 1]

#rect = [world, rect1]
rect = [world, rect1, rect2, rect3, rect4, rect5]

# Define circles here
# radius, centre x, centre y, fill
#circle1 = [0.3, 0.3, 0.2, 1]

''' Cluttered world stuff: '''
circle1 = [0.65, 0.06, -0.3, 1]

circle = [circle1]#, circle2]


# Create a white image
#img = 255*np.ones((size_px,size_px,3), np.uint8)
img = np.zeros((size_px,size_px,3), np.uint8)

# Create the world


# plot rectangles
for r in rect:
    rh = r[0]
    rv = r[1]
    rx = r[2]
    ry = r[3]
    theta = r[4] * np.pi / 180
    fill = r[5]

    rot = np.array(((np.cos(theta), -np.sin(theta)),
                    (np.sin(theta), np.cos(theta))))

    corners = np.transpose(np.array([[rh/2,rv/2],[-rh/2,rv/2],[-rh/2,-rv/2],[rh/2,-rv/2]]))
    corners = np.transpose(np.matmul(rot, corners)) + np.array([[rx,ry],[rx,ry],[rx,ry],[rx,ry]])
    corners = np.true_divide(corners, resolution_m)

    corners[:, 1] *= -1
    corners += np.array([[size_px/2,size_px/2],[size_px/2,size_px/2],[size_px/2,size_px/2],[size_px/2,size_px/2]])
    corners = corners.astype(np.int32)

    corners = corners.reshape((-1,1,2))

    if fill == 1:
        cv2.fillPoly(img, [corners], 0)
        cv2.polylines(img, [corners], True, (0, 0, 0), draw_thickness)
    elif fill == -1:
        cv2.fillPoly(img, [corners], (255, 255, 255))
    else:
        cv2.polylines(img, [corners], True, (0, 0, 0), draw_thickness)

# plot circles
for c in circle:
    cr = c[0]
    ch = c[1]
    cv = c[2]
    fill = c[3]

    centre = np.array([ch/resolution_m + size_px/2,-cv/resolution_m + size_px/2]).astype(np.int32)
    radius = int(cr/resolution_m)

    if fill == 1:
        cv2.circle(img, tuple(centre), radius, (0, 0, 0), -1)
    cv2.circle(img, tuple(centre), radius, (0, 0, 0), draw_thickness)
    print(centre, radius)

# Sanity check for any points
#cv2.circle(img, (629, 113), radius=10, color=(0, 0, 255), thickness=-1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

path = os.path.abspath(os.getcwd())

if inflation_m > 0.2:
    path = os.path.join(path, "cluttered_world/map_thick.png")
else:
    path = os.path.join(path, "cluttered_world/map_thin.png")

cv2.imwrite(path, img)