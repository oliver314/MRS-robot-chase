#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy
import cv2
import os

from robots import baddie, police_car
from utils import get_repulsive_field_from_obstacles, normalize, Particle, update_particles

from sensor_msgs.msg import PointCloud

DISTANCE_CONSIDERED_CAUGHT = 0.25
WALL_OFFSET = 4.
CYLINDER_POSITIONS = np.array([[.3, .2]], dtype=np.float32)
CYLINDER_RADIUSS = [.3]

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

# Lidar cutoff
lidar_cutoff = 3

# --------------------------- CONTROL METHODS ------------------------

##############################################
#### Simple method (random + closest) ########
##############################################

def baddies_random_movement_method(baddies, police):
  '''every baddie performs obstacle avoidance by using a braitenberg controller based on its scan measurements
  '''
  def braitenberg_controller(front, front_left, front_right, left, right):
    u = 1.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.
    # take inverse: make a very small input stand out, do not care about large inputs
    input_data = np.transpose([1/left, 1/front_left, 1/front, 1/front_right, 1/right, 1])
    w1 = np.array([ 0, -0.5, -1, -0.5,  0, 3], dtype=float)
    w2 = np.array([-1,   -1,  0,    1,  1, 0], dtype=float)
    u = np.tanh(w1.dot(input_data))
    w = 0.6 * np.tanh(w2.dot(input_data))
    return u, w

  for baddie in baddies:
      u, w = braitenberg_controller(*baddie.scan())
      baddie.set_vel(u, w)

def police_closest_method(police, baddies):
  '''every police car follows the baddie it is closest to'''
  print("Note police cars have no obstacle avoidance if the closest method is used")
  for police_car in police:
    closest_baddie = None 
    min_distance = np.inf
    for baddie in baddies:
      if np.linalg.norm(police_car.pose[:2] - baddie.pose[:2]) < min_distance and not baddie.caught:
        min_distance = np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])
        closest_baddie = baddie
    police_car.set_vel_holonomic(*(-police_car.pose[:2] + closest_baddie.pose[:2]))
    
##############################################
#### Potential field method ##################
##############################################

def baddies_pot_field_method(baddies, police):
  '''potential field method for baddies: assumes baddies know location of police'''
  P_gain_repulsive = 3
  rep_cutoff_distance = 2
  for baddie in baddies:
    # Have police cars as obstacles
    obstacle_positions = np.append(CYLINDER_POSITIONS, np.array([police_car.pose[:2] for police_car in police]), 0)
    obstacle_radii = np.append(CYLINDER_RADIUSS, [0.01 for police_car in police], 0)

    v = get_repulsive_field_from_obstacles(baddie.pose[:2], P_gain_repulsive, rep_cutoff_distance, WALL_OFFSET, obstacle_positions, obstacle_radii)
    
    baddie.set_vel_holonomic(*v)


def police_pot_field_method(police, baddies):
  '''potential field method for police'''
  P_gain_repulsive = 1
  rep_cutoff_distance = 1
  P_gain = 2

  for police_car in police:
    obstacle_positions = CYLINDER_POSITIONS
    obstacle_radii = CYLINDER_RADIUSS
    for police_car_adj in police:
      if police_car_adj != police_car:
        obstacle_positions = np.append(obstacle_positions, np.array([police_car_adj.pose[:2]]), 0)
        obstacle_radii = np.append(obstacle_radii, [0.01], 0)
    v = get_repulsive_field_from_obstacles(police_car.pose[:2], P_gain_repulsive, rep_cutoff_distance, WALL_OFFSET, obstacle_positions, obstacle_radii)

    # attractive field:
    for baddie in baddies:
      if not baddie.caught:
        v += - P_gain * normalize(police_car.pose[:2] - baddie.pose[:2])
        break
    police_car.set_vel_holonomic(*v)


##############################################
#### Testing for estimation stuff ############
##############################################

def baddies_est_test_method(baddies, police):
  pass

def police_est_test_method(police, baddies):
  # move a litte bit
  for police_car in police:
    if police_car.pose[0]<-0.6:
      police_car.set_vel(0.1, 0)
    else:
      police_car.set_vel(0, 0)


def baddies_from_lidar(police):
  xy_raw1 = police[0].lidar_xy()
  xy_raw2 = police[1].lidar_xy()
  xy_raw3 = police[2].lidar_xy()

  xy_raw = np.vstack((xy_raw1, xy_raw2, xy_raw3))
  #xy_raw = xy_raw1
  #police[0].set_vel(0.1, 0)
  #print("lidar scan", police[0].lidar_scan()[0])
  #print("lidar xy", xy_raw[0])

  xy = np.true_divide(xy_raw, resolution_m)
  xy[:, 1] *= -1
  xy += size_px/2
  xy = xy.astype(np.int32)

  robot_PC = []
  for i, coord in enumerate(xy):
    if not np.any((coord < 0)|(coord > size_px)) and map_img[coord[1]][coord[0]] == 255:
      robot_PC.append(xy_raw[i])
  robot_PC = np.asarray(robot_PC)

  identified = []
  for i in range(len(robot_PC)):
    if len(identified) == 0:
      identified.append(robot_PC[i])
    else:
      flag = False
      for j in range(i):
        if np.linalg.norm(robot_PC[i] - robot_PC[j]) < 0.25:
          flag = True

      if not flag:
        identified.append(robot_PC[i])

  # Exclude any positions that corresponds to your fellow officers from the list
  baddies_identified = []
  for est in identified:
    is_police = False
    for police_car in police:
      if np.linalg.norm(police_car.pose[:2] - est) < 0.2:
        is_police = True
        break

    if not is_police:
      baddies_identified.append(est)

  #print(xy_raw)
  #print(np.asarray(identified))
  print(np.asarray(baddies_identified))
  print("\n\n")

  return baddies_identified


# ----------------------------SUPPORT FUNCTIONS ----------------------------

def check_if_any_caught(police, baddies):
  for police_car in police:
    for baddie in baddies:
      if np.linalg.norm(police_car.pose[:2] - baddie.pose[:2]) < DISTANCE_CONSIDERED_CAUGHT:
        baddie.caught = True

def check_if_all_caught(police, baddies):
  for baddie in baddies:
    if not baddie.caught:
      return False
  # if we get here, all baddies caught! Stop the police pro forma
  for police_car in police:
    police_car.set_vel_holonomic(0, 0)
  for baddie in baddies:
    baddie.set_vel_holonomic(0, 0)
  return True

# ----------------------------MAIN FUNCTION ----------------------------

def run(args):
  rospy.init_node('robot_chase')
  if args.mode_baddies == "random":
    baddies_method = baddies_random_movement_method
  elif args.mode_baddies == "potential_field":
    baddies_method = baddies_pot_field_method
  elif args.mode_baddies == "est_test":
    baddies_method = baddies_est_test_method
  else:
    raise NotImplementedError("%s not implemented" % args.mode_baddies)

  if args.mode_police == "closest":
    police_method = police_closest_method
  elif args.mode_police == "potential_field":
    police_method = police_pot_field_method
  elif args.mode_police == "est_test":
    police_method = police_est_test_method
  else:
    raise NotImplementedError("%s not implemented" % args.mode_police)

  # Particles init
  particle_publisher = rospy.Publisher('/particles', PointCloud, queue_size=1)
  num_particles = 200
  particles = [Particle(xlim, ylim, map_img, lidar_cutoff) for _ in range(num_particles)]

  for i, p in enumerate(particles):
    p.set_pose(1.5, (i+1)*0.5)
    if i == 2:
      break
  nr_baddies = args.nr_baddies
  nr_police = args.nr_police

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  idx = 0

  police_names = ["police" + str(i+1) for i in range(int(nr_police))]
  baddies_names = ["baddie" + str(i+1) for i in range(int(nr_baddies))]

  baddies = [baddie(name) for name in baddies_names]
  police = [police_car(name) for name in police_names]

  frame_id = 0
  while not rospy.is_shutdown():

    baddies_method(baddies, police)
    police_method(police, baddies)

    check_if_any_caught(police, baddies)

    if check_if_all_caught(police, baddies):
      print("Done in iteration %d" % idx)
      break
    idx += 1

    baddies_list = baddies_from_lidar(police)
    # Move, compute weight, resample particles
    particles, particle_msg = update_particles(particles, num_particles, frame_id, police, baddies_list)

    # publish particles
    particle_publisher.publish(particle_msg)

    rate_limiter.sleep()
    frame_id += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs robot chase')
  parser.add_argument('--mode_baddies', action='store', default='random', help='Method.', choices=['random', 'potential_field', 'est_test'])
  parser.add_argument('--mode_police', action='store', default='closest', help='Method.', choices=['closest', 'potential_field', 'est_test'])
  parser.add_argument('--nr_baddies', action='store', default=3)
  parser.add_argument('--nr_police', action='store', default=3)
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
