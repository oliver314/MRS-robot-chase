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
from particles_utils import Particle, get_baddies_estimation
from control_functions import baddies_random_movement_method, baddies_pot_field_method, police_closest_method, \
								police_pot_field_method, police_closest_rrt_method, police_rrt_all_on_one_method, \
								police_rrt_one_on_one_method, police_rrt_zone_method

from sensor_msgs.msg import PointCloud

DISTANCE_CONSIDERED_CAUGHT = 0.25
WALL_OFFSET = 4.
CYLINDER_POSITIONS = np.array([[.3, .2]], dtype=np.float32)
CYLINDER_RADIUSS = [.3]

# Map occupancy information
size_m = 10		 # 10m x 10m size
resolution_m = 0.01 # 1cm resolution
MAP_NAME = "simple_world_big"
#MAP_NAME = "cluttered_world"

# Limits (in meters) for x and y in the environment
xlim = [-4, 4]
ylim = [-4, 4]


# load world image
path_thick = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds/'+MAP_NAME+'/map_thick.png')
path_thin = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds/'+MAP_NAME+'/map_thin.png')
map_thick_img = cv2.imread(path_thick, 0)
map_thin_img = cv2.imread(path_thin, 0)


# ----------------------------SUPPORT FUNCTIONS ----------------------------

def check_if_any_caught(police, baddies):
	for police_car in police:
		for baddie in baddies:
			if not baddie.caught and np.linalg.norm(police_car.pose[:2] - baddie.gt_pose[:2]) < DISTANCE_CONSIDERED_CAUGHT:
				baddie.caught = True
				baddie.temp_caught_flag = True
				baddie.set_vel_holonomic(0,0)


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

	map_img = map_thick_img
	if mode_estimator == 'line_of_sight':
		map_img = map_thin_img

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
	prev_baddie_measurement = [None] * int(nr_baddies)

	# Live plot of lidar measurements if you are using lidar to estimate baddies positions
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

	# Setup pose recording for plotting
	pose_history = []
	with open('/tmp/gazebo_exercise.txt', 'w'):
		pass

	# Main loop
	while not rospy.is_shutdown():
		#baddies_method(baddies, police)
		#police_method(police, baddies)

		check_if_any_caught(police, baddies)

		if mode_estimator != 'gt':
			# Updates the pose estimation of baddies
			prev_baddie_measurement = get_baddies_estimation(police, baddies, prev_baddie_measurement, mode_estimator=='line_of_sight', map_img,
													   particles, particle_publisher, num_particles, idx, live_plot)

		# Print baddie status for debugging purposes
		#for i in range(len(baddies)):
		#  print(baddies[i].name," - pose: ", baddies[i].pose, " caught: ", baddies[i].caught)
		#print("\n\n")

		if check_if_all_caught(police, baddies):
			print("Done in iteration %d" % idx)
			break
		idx += 1

		# logging here
		#pose_history.append(np.concatenate([groundtruth.pose, absolute_point_position], axis=0))
		pose_history.append([baddies[0].gt_pose])
		if len(pose_history) % 10:
			with open('/tmp/gazebo_exercise.txt', 'a') as fp:
				fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
				pose_history = []
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