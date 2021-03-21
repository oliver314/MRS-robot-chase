from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np


WALL_OFFSET = 2.
CYLINDER_POSITIONS = np.array([[.3, .2]], dtype=np.float32)
#CYLINDER_POSITIONS = np.array([[0., 0.]], dtype=np.float32) # Question d)
CYLINDER_RADIUSS = [.3]
#CYLINDER_POSITIONS = np.array([[0.5, 0.], [0., 0.5]], dtype=np.float32) # Question f) Solution: add virtual obstacles at local minima
#CYLINDER_RADIUSS = [.3, .3]
# CYLINDER_POSITIONS = np.array([[-0.75, 0.5], [0.5, 0.], [0., 0.5], [0.5, -0.75]], dtype=np.float32) # Question f) harder
# CYLINDER_RADIUSS = [.3, .3, .3, .3 ]
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSITION = np.array([-1.5, -1.5], dtype=np.float32)
MAX_SPEED = .5


def get_velocity_to_reach_goal(position, goal_position):
  '''Potential field U(x,y) = k/2 (dx^2 + dy^2)
  '''
  a = 1
  v = - a * (position - goal_position)
  return v


def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
  '''Compute the velocity field needed to avoid the obstacles
  In the worst case there might a large force pushing towards the
  obstacles (consider what is the largest force resulting from the
  get_velocity_to_reach_goal function). 
  PIAZZA says ignore statement: Make sure to not create speeds that are larger than max_speed for each obstacle. 
  Both obstacle_positions and obstacle_radii are lists.
  '''

  v = np.zeros(2, dtype=np.float32)
  a = 1
  b = 0.5

  # Method 1 uses an exponential decay: Advantage: can tune decay constant
  '''a = 10
  b = 3
  for obstacle_pos, obstacle_rad in zip(obstacle_positions, obstacle_radii):
    delta_x_vector = position - obstacle_pos
    distance = np.linalg.norm(delta_x_vector)
    v_cand = normalize(delta_x_vector) * np.exp(-b*(distance - obstacle_rad)) * a
    v += v_cand'''

  # Method 2: Has more of a basis as a potential field
  for obstacle_pos, obstacle_rad in zip(obstacle_positions, obstacle_radii):
    delta_x_vector_to_center = position - obstacle_pos
    delta_x_vector = delta_x_vector_to_center - obstacle_rad * normalize(delta_x_vector_to_center)
    distance = np.linalg.norm(delta_x_vector)
    v_cand = - a*(1./b - 1/distance) * 1/distance**2 * delta_x_vector
    if distance < b:
      v += v_cand

  return v


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(
      position,
      CYLINDER_POSITIONS,
      CYLINDER_RADIUSS)
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid
  return cap(v, max_speed=MAX_SPEED) + 0.001*np.random.randn(2) # Question e)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots()
  # Plot field.
  X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')

  # Plot environment.
  for pos, rad in zip(CYLINDER_POSITIONS, CYLINDER_RADIUSS):
    ax.add_artist(plt.Circle(pos, rad, color='gray'))
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.01
  x = START_POSITION
  positions = [x]
  number_of_seconds_still_before_declare_local = 0.5
  least_time_before_new_local = 0
  last_local_at = 0
  for t in np.arange(0., 20., dt):
    # Question 1f
    if len(positions) > number_of_seconds_still_before_declare_local/dt \
       and np.linalg.norm(positions[-int(number_of_seconds_still_before_declare_local//dt)] - x) < 0.001 \
       and np.linalg.norm(GOAL_POSITION - x) > 0.5 and len(positions) > last_local_at + least_time_before_new_local/dt:
      CYLINDER_POSITIONS = np.concatenate((CYLINDER_POSITIONS, [x + v]), axis=0)
      CYLINDER_RADIUSS.append(0)
      print(CYLINDER_POSITIONS)
      print(CYLINDER_RADIUSS)
      last_local_at = len(positions)
      #plt.scatter(x[0], x[1], lw=6, c='g')
    v = get_velocity(x, args.mode)
    x = x + v * dt
    positions.append(x)
  positions = np.array(positions)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.show()
