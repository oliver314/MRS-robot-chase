from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml
import cv2
import threading


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 500


def sample_random_position(occupancy_grid, goal_position):
  ''' Sample a valid random position. 
  The corresponding cell must be free in the occupancy grid.
  '''
  position = np.zeros(2, dtype=np.float32)
  while True:
    mode = np.random.rand()
    if mode > 0.3:
      # choose randomly in field
      pos_x = np.random.randint(0, occupancy_grid.values.shape[0], 1)[0]
      pos_y = np.random.randint(0, occupancy_grid.values.shape[1], 1)[0]
      position = occupancy_grid.get_position(pos_x, pos_y)
    elif mode > 0:
      # choose close to goal
      position = goal_position + np.random.normal([0, 0], 1.5, 2)
    else:
      # choose close to obstacle (would need to be a bit more dynamic for new maps etc.)
      position = np.random.normal([0.3, 0.2], 0.8, 2)
    if occupancy_grid.is_free(position):
      return position


def adjust_pose(node, final_position, occupancy_grid):
  '''Check whether there exists a simple path that links node.pose
  to final_position. This function needs to return a new node that has
  the same position as final_position and a valid yaw. The yaw is such that
  there exists an arc of a circle that passes through node.pose and the
  adjusted final pose. If no such arc exists (e.g., collision) return None.
  Assume that the robot always goes forward.
  '''

  # Calculate center of circle deterministically
  x_1, y_1, x_2, y_2 = node.pose[0], node.pose[1], final_position[0], final_position[1] 
  if node.direction[1] == 0:
    a = 1e9
  else:
    a = -node.direction[0]/node.direction[1]

  x_center = (0.5*(x_2**2-x_1**2) + a*x_1*(y_2-y_1)+0.5*(y_2-y_1)**2)/(a*(y_2-y_1)+x_2-x_1)
  y_center = a*x_center - a*x_1 + y_1

  #Generate yaw given the center. Check direction
  yaw = np.arctan2(x_2 - x_center, y_center - y_2)
  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  final_pose[2] = yaw
  final_node = Node(final_pose)
  # Assuming we ll never do more than 180 deg turn in one go: valid given how parents are chosen
  if np.dot(final_node.direction, node.direction) < 0:
    yaw += np.pi
    final_pose[2] = yaw
    final_node = Node(final_pose)

  # Doublecheck
  center, radius = find_circle(node, final_node)
  #assert np.linalg.norm(np.array([x_center, y_center]) - center) < 1e-1


  # Check whether it goes through an obstacle
  theta1 = np.arctan2(y_1 - center[1], x_1 - center[0])
  theta2 = np.arctan2(y_2 - center[1], x_2 - center[0])

  theta = min(theta1, theta2)
  theta_end = max(theta1, theta2)
  if theta_end - theta > np.pi:
    theta_end = theta + 2*np.pi
    theta = max(theta1, theta2)

  while theta < theta_end:
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    position = [x, y]
    if not occupancy_grid.is_free(position):
      return None
    theta += 0.01/radius

  
  return final_node


# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c



def rrt(start_pose, goal_position, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  v = None
  for _ in range(MAX_ITERATIONS): 
    position = sample_random_position(occupancy_grid, goal_position)
    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position
    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    potential_parent = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
    # Pick a node at least some distance away but not too far.
    # We also verify that the angles are aligned (within pi / 4).
    u = None
    for n, d in potential_parent:
      if d > .2 and d < 1.5 and n.direction.dot(position - n.position) / d > 0.70710678118:
        u = n
        break
    else:
      continue
    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)
    if np.linalg.norm(v.position - goal_position) < .2:
      final_node = v
      break

  return start_node, final_node



def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.



def get_path(final_node):
  # Construct path from RRT solution.
  if final_node is None:
    return []
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  # Put a point every 5 cm.
  distance = 0.05
  offset = 0.
  points_x = []
  points_y = []
  for u, v in zip(path, path[1:]):
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    clockwise = np.cross(u.direction, du).item() > 0.
    # Generate a point every 5cm apart.
    da = distance / radius
    offset_a = offset / radius
    if clockwise:
      da = -da
      offset_a = -offset_a
      if theta2 > theta1:
        theta2 -= 2. * np.pi
    else:
      if theta2 < theta1:
        theta2 += 2. * np.pi
    angles = np.arange(theta1 + offset_a, theta2, da)
    offset = distance - (theta2 - angles[-1]) * radius
    points_x.extend(center[X] + np.cos(angles) * radius)
    points_y.extend(center[Y] + np.sin(angles) * radius)
  return zip(points_x, points_y)
  


def get_velocity(position, path_points):
  '''Return the velocity needed to follow the
  path defined by path_points. Assume holonomicity of the
  point located at position.
  '''
  v = np.zeros_like(position)
  if len(path_points) == 0:
    return v
  # Stop moving if the goal is reached.
  if np.linalg.norm(position - path_points[-1]) < .2:
    return v

  # Determine which path point is closest
  min_distance = np.inf
  closest_point = None
  for i, path_point in enumerate(path_points):
    if np.linalg.norm(position - path_point) < min_distance:
      closest_point_index = i
      min_distance = np.linalg.norm(position - path_point)

  # Determine whether the closest pathpoint is behind or in front of the robot by considering the second closest
  # Then choose the point ahead of the next one as target
  if len(path_points) == 1:
    delta = path_points[0] - position
  elif closest_point_index == 0:
    delta = path_points[1] - position
  elif closest_point_index >= len(path_points)-2:
    delta = path_points[-1] - position
  elif np.linalg.norm(position - path_points[closest_point_index+1]) > np.linalg.norm(position - path_points[closest_point_index-1]):
    delta = path_points[closest_point_index+1] - position
  else:
    delta = path_points[closest_point_index+2] - position

  # Proportional control of holonomic point
  k = 1
  return k * delta/np.linalg.norm(delta)



def draw_solution(start_node, final_node=None):
  ax = plt.gca()
  ax.set_xlim((-4, 4))
  ax.set_ylim((-4, 4))

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                             theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                             color=color, lw=lw))

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent



def re_evaluate_path_thread(start_pose, target_pose, police_car, current_path, called_idx, occupancy_grid):
    start_node, final_node = rrt(start_pose, target_pose, occupancy_grid)
    if final_node is None:
      final_node = Node(np.array([target_pose[0], target_pose[1], 0]))
      final_node.parent = Node(start_pose)

    current_path[police_car.name] = get_path(final_node)
    called_idx[police_car.name] = 0
    print("Updating trajectory with target " + str(target_pose) + " for " + police_car.name)

class rrt_wrapper:
  def __init__(self, size_px, resolution, map_name="simple_world_big"):
    # Load map.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds/'+map_name+'/map.png')
    img = cv2.imread(path, 0)
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = UNKNOWN
    occupancy_grid[img < .1*255] = OCCUPIED
    occupancy_grid[img > .9*255] = FREE
    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]
    self.occupancy_grid = OccupancyGrid(occupancy_grid, [-size_px//2, -size_px//2, 0], resolution)
    #self.occupancy_grid.draw()
    self.current_path = {}
    self.called_idx = {}
    self.rrt_thread = {}
    #plt.ion()
    #plt.show()


  def rrt_next_vel(self, police_car, target_pose):
      start_pose = police_car.pose
      if police_car.name not in self.current_path or self.called_idx[police_car.name] > 10:
        # Run RRT.
        plt.clf()
        #self.occupancy_grid.draw()
        if police_car.name not in self.rrt_thread or not self.rrt_thread[police_car.name].is_alive():

          #draw_solution(start_node, final_node)
          #plt.title(police_car.name)
          #plt.draw()
          #plt.pause(0.001)

          # in a thread so that velocity is continously updated, even as we recalculate trajectory (and robot doesn t execute last velocity setting during that whole time)
          self.rrt_thread[police_car.name] = threading.Thread(target=re_evaluate_path_thread, args=(start_pose, target_pose, police_car, self.current_path, self.called_idx, self.occupancy_grid))
          self.rrt_thread[police_car.name].start()
          #print("RRT thread started!")
          if police_car.name not in self.current_path or len(self.current_path[police_car.name]) == 0:
            print("And joined")
            self.rrt_thread[police_car.name].join()
        
      self.called_idx[police_car.name] += 1
      v = get_velocity(start_pose[:2], np.array(self.current_path[police_car.name], dtype=np.float32))
      #print("Update velocity!")
      return v
