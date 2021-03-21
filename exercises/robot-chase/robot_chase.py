#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

from robots import baddie, police_car


DISTANCE_CONSIDERED_CAUGHT = 0.2


# --------------------------- CONTROL METHODS ------------------------


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
  for police_car in police:
    closest_baddie = None 
    min_distance = np.inf
    for baddie in baddies:
      if np.linalg.norm(police_car.pose[:2] - baddie.pose[:2]) < min_distance and not baddie.caught:
        min_distance = np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])
        closest_baddie = baddie
      police_car.set_vel_holonomic(*(-police_car.pose[:2] + baddie.pose[:2]))
    

# ----------------------------MAIN FUNCTION ----------------------------


def check_if_any_caught(police, baddies):
  for police_car in police:
    for baddie in baddies:
      if np.linalg.norm(police_car.pose[:2] - baddie.pose[:2]) < DISTANCE_CONSIDERED_CAUGHT:
        baddie.caught = True

def check_if_all_caught(police, baddies):
  for baddie in baddies:
    if not baddie.caught:
      return False
  return True

def run(args):
  rospy.init_node('robot_chase')
  if args.mode_baddies == "random":
    baddies_method = baddies_random_movement_method
  else:
    raise NotImplementedError("%s not implemented" % args.mode_baddies)

  if args.mode_police == "closest":
    police_method = police_closest_method
  else:
    raise NotImplementedError("%s not implemented" % args.mode_police)

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  idx = 0

  police_names = ["police1", "police2", "police3"]
  baddies_names = ["baddie1", "baddie2", "baddie3"]

  baddies = [baddie(name) for name in baddies_names]
  police = [police_car(name) for name in police_names]

  while not rospy.is_shutdown():

    baddies_method(baddies, police)
    police_method(police, baddies)

    check_if_any_caught(police, baddies)

    if check_if_all_caught(police, baddies):
      print("Done in iteration %d" % idx)
      break
    idx += 1

    rate_limiter.sleep()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs robot chase')
  parser.add_argument('--mode_baddies', action='store', default='random', help='Method.', choices=['random'])
  parser.add_argument('--mode_police', action='store', default='closest', help='Method.', choices=['closest'])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
