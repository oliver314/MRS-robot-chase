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
from scipy.optimize import linear_sum_assignment as hungarian_alg
from copy import copy

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
size_m = 10		 # 10m x 10m size
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
	w2 = np.array([-1,   -1,  0,	1,  1, 0], dtype=float)
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
  global rrt_wrap
  for police_car in police:
	closest_baddie = None 
	min_distance = np.inf
	for baddie in baddies:
	  if np.linalg.norm(police_car.pose[:2] - baddie.pose[:2]) < min_distance and not baddie.caught:
		min_distance = np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])
		closest_baddie = baddie
	v = rrt_wrap.rrt_next_vel(police_car, closest_baddie.pose[:2])
	police_car.set_vel_holonomic(*v, backup=True)
	

def police_rrt_all_on_one_method(police, baddies):
  '''baddie minimising the summed distances to all police carsis selected.
  Every police car follows it, trajectory via rrt, thus now uses obstacle avoidance
  '''
  global rrt_wrap
  baddies_distances = [0]*len(baddies)
  for police_car in police:
	for i, baddie in enumerate(baddies):
	  if baddie.caught:
		  baddies_distances[i] = np.inf
	  else:
		  baddies_distances[i] += np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])
  closest_baddie = baddies[baddies_distances.index(min(baddies_distances))]
  for police_car in police:
	v = rrt_wrap.rrt_next_vel(police_car, closest_baddie.pose[:2])
	police_car.set_vel_holonomic(*v, backup=True)


def police_rrt_one_on_one_method(police, baddies):
  '''Use Hungarian algorithm to allocate a baddie to each police_car.
  '''
  global rrt_wrap
  cost_matrix = np.zeros((len(police), len(baddies)))
  for i, police_car in enumerate(police):
	for j, baddie in enumerate(baddies):
	  if not baddie.caught:
		  cost_matrix[i][j] = np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])

  police_car_idx, baddie_idx = hungarian_alg(cost_matrix)
  for idx_p, idx_b in zip(police_car_idx, baddie_idx):
	while(baddies[idx_b].caught):
		idx_b = (idx_b + 1) % len(baddies)
	v = rrt_wrap.rrt_next_vel(police[idx_p], baddies[idx_b].pose[:2])
	police[idx_p].set_vel_holonomic(*v, backup=True)


def police_rrt_zone_method(police, baddies):
	'''divide square up in 9 zones. Select one baddie and send one police car in each of the adjacent cells.
	'''
	baddies_distances = [0]*len(baddies)
	for police_car in police:
	  for i, baddie in enumerate(baddies):
		if baddie.caught:
			baddies_distances[i] = np.inf
		else:
			baddies_distances[i] += np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])
	closest_baddie = baddies[baddies_distances.index(min(baddies_distances))]

	grid_side_count = 4
	field_width = 8
	cell_width = field_width/grid_side_count

	police_in_between_distance = np.sum([np.linalg.norm(police_car.pose[:2] - police[0].pose[:2]) for police_car in police])
	if min(baddies_distances) < len(police) * cell_width and police_in_between_distance > (len(baddies)-1) * 0.7:
		print("Attack!")
  		for police_car in police:
			police_car.set_vel_holonomic(*(-police_car.pose[:2] + closest_baddie.pose[:2]))
		
	else:
		print("Encirclement")
		# create graph
		def valid(n):
			return n >= 0 and n < grid_side_count**2
		graph = [[]]*grid_side_count**2
		for i in range(grid_side_count**2):
			neighbours = []
			if valid(i+1) and (i+1) % grid_side_count != 0:
				neighbours.append(i+1)
			if valid(i-1) and i % grid_side_count != 0:
				neighbours.append(i-1)
			if valid(i+grid_side_count):
				neighbours.append(i+grid_side_count)
			if valid(i-grid_side_count):
				neighbours.append(i-grid_side_count)
			graph[i] = neighbours

		node_idx = (field_width/2 + closest_baddie.pose[0]) // cell_width
		node_idx += grid_side_count * ((field_width/2 + closest_baddie.pose[1]) // cell_width)
		node_idx = int(node_idx)

		def get_target_from_node(node):
			target = [((node % grid_side_count) + 0.5) * cell_width - field_width/2]
			target.append(((node//grid_side_count) + 0.5) * cell_width - field_width/2)
			return target

		goals = [get_target_from_node(node) for node in graph[node_idx]]
		cost_matrix = np.zeros((len(police), len(goals)))
		for i, police_car in enumerate(police):
			for j, goal in enumerate(goals):
				  cost_matrix[i][j] = np.linalg.norm(police_car.pose[:2] - goal)
	   
		police_car_idx, goal_idx = hungarian_alg(cost_matrix)
		for idx_p, idx_b in zip(police_car_idx, goal_idx):
			v = rrt_wrap.rrt_next_vel(police[idx_p], goals[idx_b])
			police[idx_p].set_vel_holonomic(*v, backup=True)

		cost = cost_matrix[police_car_idx, goal_idx].sum()
		# if everyone in place, get remaining robot closer so that attack can start!
		if cost < 0.5:
			for i, police_car in enumerate(police):
				if i not in police_car_idx:
					police_car.set_vel_holonomic(*(-police_car.pose[:2] + closest_baddie.pose[:2]))

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
  else:
	raise NotImplementedError("%s not implemented" % args.mode_baddies)

  if args.mode_police == "closest":
	police_method = police_closest_method
  elif args.mode_police == "closest_rrt":
	police_method = police_closest_rrt_method
  elif args.mode_police == "potential_field":
	police_method = police_pot_field_method
  elif args.mode_police == "rrt_all_on_one":
	police_method = police_rrt_all_on_one_method
  elif args.mode_police == "rrt_one_on_one":
	police_method = police_rrt_one_on_one_method
  elif args.mode_police == "rrt_zone":
	police_method = police_rrt_zone_method
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

  live_plot = False

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

	if check_if_all_caught(police, baddies):
	  print("Done in iteration %d" % idx)
	  break
	idx += 1

	rate_limiter.sleep()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs robot chase')
  parser.add_argument('--mode_baddies', action='store', default='random', help='Method.', choices=['random', 'potential_field', 'est_test'])
  parser.add_argument('--mode_police', action='store', default='closest', help='Method.')
  parser.add_argument('--nr_baddies', action='store', default=3)
  parser.add_argument('--nr_police', action='store', default=3)
  parser.add_argument('--mode_estimator', action='store', default='gt', help='Gt - police has access to gt baddie pose; line_of_sight - use geometrical line_of_sight; lidar - use lidar', choices = ['gt', 'line_of_sight', 'lidar'])
  args, unknown = parser.parse_known_args()
  try:
	run(args)
  except rospy.ROSInterruptException:
	pass
