#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy
import cv2
import os
import matplotlib.pyplot as plt

from robots import baddie, police_car
from utils import get_repulsive_field_from_obstacles, normalize
from particles_utils import Particle, get_baddies_estimation

from sensor_msgs.msg import PointCloud
from rrt import rrt_wrapper

DISTANCE_CONSIDERED_CAUGHT = 0.25
WALL_OFFSET = 4.
CYLINDER_POSITIONS = np.array([[.3, .2]], dtype=np.float32)
CYLINDER_RADIUSS = [.3]

# Map occupancy information
size_m = 10         # 10m x 10m size
resolution_m = 0.01 # 1cm resolution
size_px = int(size_m/resolution_m) # image is of size: size_px x size_px
MAP_NAME = "simple_world_big"

# Limits (in meters) for x and y in the environment
xlim = [-4, 4]
ylim = [-4, 4]


# load world image
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds/'+MAP_NAME+'/map.png')
map_img = cv2.imread(path, 0)


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

rrt_wrap = rrt_wrapper(size_m, resolution_m, MAP_NAME)
def police_closest_rrt_method(police, baddies):
  '''every police car follows the baddie it is closest to, trajectory via rrt, thus now uses obstacle avoidance'''
  for police_car in police:
    closest_baddie = None 
    min_distance = np.inf
    for baddie in baddies:
      if np.linalg.norm(police_car.pose[:2] - baddie.pose[:2]) < min_distance and not baddie.caught:
        min_distance = np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])
        closest_baddie = baddie
    global rrt_wrap
    v = rrt_wrap.rrt_next_vel(police_car, closest_baddie.pose[:2])
    police_car.set_vel_holonomic(*v)
    
##############################################
#### Potential field method ##################
##############################################

def baddies_pot_field_method(baddies, police):
  '''potential field method for baddies: assumes baddies know location of police'''
  P_gain_repulsive = 2.5
  rep_cutoff_distance = 2
  for baddie in baddies:
    # Have police cars as obstacles
    obstacle_positions = np.append(CYLINDER_POSITIONS, np.array([police_car.pose[:2] for police_car in police]), 0)
    obstacle_radii = np.append(CYLINDER_RADIUSS, [0.01 for police_car in police], 0)

    v = get_repulsive_field_from_obstacles(baddie.gt_pose[:2], P_gain_repulsive, rep_cutoff_distance, WALL_OFFSET, obstacle_positions, obstacle_radii)
    
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


# ----------------------------SUPPORT FUNCTIONS ----------------------------

def check_if_any_caught(police, baddies):
  for police_car in police:
    for baddie in baddies:
      if np.linalg.norm(police_car.pose[:2] - baddie.gt_pose[:2]) < DISTANCE_CONSIDERED_CAUGHT:
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
  else:
    raise NotImplementedError("%s not implemented" % args.mode_baddies)

  if args.mode_police == "closest":
    police_method = police_closest_method
  elif args.mode_police == "closest_rrt":
    police_method = police_closest_rrt_method
  elif args.mode_police == "potential_field":
    police_method = police_pot_field_method
  else:
    raise NotImplementedError("%s not implemented" % args.mode_police)

  nr_baddies = args.nr_baddies
  nr_police = args.nr_police
  mode_estimator = args.mode_estimator


  ##### Particles and estimation - Setup START

  # number of particles assigned to each baddie
  num_particles = 20
  # init the list of particle publishers and particles for each baddie
  particle_publisher = []
  particles = []
  for i in range(int(nr_baddies)):
    particle_publisher.append(rospy.Publisher('/particles'+str(i+1), PointCloud, queue_size=1))
    particles.append([Particle(xlim, ylim, map_img) for _ in range(num_particles)])

  # init baddies measured position list
  prev_baddie_measurement = [None, None, None]

  live_plot = True

  ##### Particles and estimation - Setup END


  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  idx = 0

  police_names = ["police" + str(i+1) for i in range(int(nr_police))]
  baddies_names = ["baddie" + str(i+1) for i in range(int(nr_baddies))]

  baddies = [baddie(name, mode_estimator=='gt') for name in baddies_names]
  police = [police_car(name) for name in police_names]

  # live plotting setup
  if live_plot:
    plt.ion()
    plt.show()


  # Main loop
  while not rospy.is_shutdown():

    baddies_method(baddies, police)
    police_method(police, baddies)

    check_if_any_caught(police, baddies)

    if mode_estimator != 'gt':
      # Updates the pose estimation of baddies
      prev_baddie_measurement = get_baddies_estimation(police, baddies, prev_baddie_measurement, mode_estimator=='line_of_sight', map_img,
                                                       particles, particle_publisher, num_particles, idx, live_plot)

    # Print baddie positions
    #print("baddie1: ", baddies[0].pose)
    #print("baddie2: ", baddies[1].pose)
    #print("baddie3: ", baddies[2].pose)
    #print("\n\n")

    if check_if_all_caught(police, baddies):
      print("Done in iteration %d" % idx)
      break
    idx += 1

    rate_limiter.sleep()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs robot chase')
  parser.add_argument('--mode_baddies', action='store', default='random', help='Method.', choices=['random', 'potential_field', 'est_test'])
  parser.add_argument('--mode_police', action='store', default='closest', help='Method.', choices=['closest', 'closest_rrt', 'potential_field', 'est_test'])
  parser.add_argument('--nr_baddies', action='store', default=3)
  parser.add_argument('--nr_police', action='store', default=3)
  parser.add_argument('--mode_estimator', action='store', default='gt', help='Gt - police has access to gt baddie pose; line_of_sight - use geometrical line_of_sight; lidar - use lidar', choices = ['gt', 'line_of_sight', 'lidar'])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
