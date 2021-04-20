from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sensor_msgs.msg import LaserScan

# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
import rospy

# Map occupancy information
size_m = 10         # 10m x 10m size
resolution_m = 0.01 # 1cm resolution
size_px = int(size_m/resolution_m) # image is of size: size_px x size_px


class SimpleLaser(object):
  def __init__(self, name, angles):
    rospy.Subscriber('/%s/scan' % name, LaserScan, self.callback)
    self._angles = angles
    #self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None

  def callback(self, msg):
    # Helper for angles.
    def _within(x, a, b):
      pi2 = np.pi * 2.
      x %= pi2
      a %= pi2
      b %= pi2
      if a < b:
        return a <= x and x <= b
      return a <= x or x <= b;

    # Compute indices the first time.
    if self._indices is None:
      self._indices = [[] for _ in range(len(self._angles))]
      for i, d in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        for j, center_angle in enumerate(self._angles):
          if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
            self._indices[j].append(i)
      #print(self._indices)
    ranges = np.array(msg.ranges)
    for i, idx in enumerate(self._indices):
      # We do not take the minimum range of the cone but the 10-th percentile for robustness.
      self._measurements[i] = np.percentile(ranges[idx], 10)


  @property
  def ready(self):
    return not np.isnan(self._measurements[0])

  @property
  def measurements(self):
    return self._measurements

class GroundtruthPose(object):
  def __init__(self, name):
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._name = name

  def callback(self, msg):
    idx = [i for i, n in enumerate(msg.name) if n == self._name]
    if not idx:
      raise ValueError('Specified name "{}" does not exist.'.format(self._name))
    idx = idx[0]
    self._pose[0] = msg.pose[idx].position.x
    self._pose[1] = msg.pose[idx].position.y
    _, _, yaw = euler_from_quaternion([
        msg.pose[idx].orientation.x,
        msg.pose[idx].orientation.y,
        msg.pose[idx].orientation.z,
        msg.pose[idx].orientation.w])
    self._pose[2] = yaw

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose


X = 0
Y = 1
YAW = 2


def feedback_linearized(pose, velocity, epsilon):
  '''feedback-linearization to follow the velocity
  vector given as argument. Epsilon corresponds to the distance of
  linearized point in front of the robot.
  '''
  u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW]) #m/s
  w = (1/epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW])) # [rad/s] going counter-clockwise.
  return u, w

def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n

def get_repulsive_field_from_obstacles(position, a, b, WALL_OFFSET, obstacle_positions, obstacle_radii):

    v = np.zeros(2, dtype=np.float32)

    def adapt_v(delta_x_vector, v):
      distance = np.linalg.norm(delta_x_vector)
      v_cand = - a*(1./b - 1/distance) * 1/distance**2 * delta_x_vector
      if distance < b:
        v = v + v_cand
      return v

    for obstacle_pos, obstacle_rad in zip(obstacle_positions, obstacle_radii):
      delta_x_vector_to_center = position - obstacle_pos
      delta_x_vector = delta_x_vector_to_center - obstacle_rad * normalize(delta_x_vector_to_center)
      v = adapt_v(delta_x_vector, v)

    delta_x_vector_to_wall = position - np.array([position[0], WALL_OFFSET])
    v = adapt_v(delta_x_vector_to_wall, v)
    delta_x_vector_to_wall = position - np.array([position[0], -WALL_OFFSET])
    v = adapt_v(delta_x_vector_to_wall, v)
    delta_x_vector_to_wall = position - np.array([WALL_OFFSET, position[1]])
    v = adapt_v(delta_x_vector_to_wall, v)
    delta_x_vector_to_wall = position - np.array([-WALL_OFFSET, position[1]])
    v = adapt_v(delta_x_vector_to_wall, v)
    
    return v

def logging_function(baddies, police, idx):

  output = [idx]
  for baddie in baddies:
    min_error = 100
    for baddie2 in baddies:
      error = np.linalg.norm(baddie2.gt_pose[:2] - baddie.pose[:2])
      if error < min_error:
        min_error = error
    det_var = np.linalg.det(baddie.est_variance)
    output.append(min_error)
    output.append(det_var)
    output.append(baddie.est_variance[0][0]**2 + baddie.est_variance[1][1]**2)
  return output
  #[idx, b1.gt_pose[0], b1.gt_pose[1], b1.pose[0], b1.pose[1],
  #      b1.est_variance[0][0], b1.est_variance[0][1], b1.est_variance[1][0], b1.est_variance[1][1]]