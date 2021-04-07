from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import copy

# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Point32

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

class Particle(object):
  """Represents a particle."""

  def __init__(self, xlim, ylim, map_img, dist_cutoff):
    self._pose = np.zeros(3, dtype=np.float32)
    self._weight = 1.
    self.xlim = xlim
    self.ylim = ylim
    self.map = map_img
    self.dist_cutoff = dist_cutoff
    # Choose a random position on the pose for the particles
    self.sample_random_pose()

  def set_pose(self, x, y):
    self._pose[0] = x
    self._pose[1] = y

  # Set a random position
  def sample_random_pose(self):
    xrng = abs(self.xlim[1] - self.xlim[0])
    xmin = min(self.xlim[1] , self.xlim[0])

    yrng = abs(self.ylim[1] - self.ylim[0])
    ymin = min(self.ylim[1] , self.ylim[0])

    while 1:
      self._pose[0] = xrng * np.random.random() + xmin
      self._pose[1] = yrng * np.random.random() + ymin
      self._pose[2] = 2 * np.pi * np.random.rand() - np.pi

      if self.is_valid(self._pose):
        break

  # Returns true if a position of a particle is within bounds of the environment
  def is_valid(self, position):
    valid = False

    x = position[0]
    y = position[1]

    pose = np.array([( x / resolution_m) + size_px / 2, (-y / resolution_m) + size_px / 2])
    pose = pose.astype(np.int32)

    # Check if that location is valid
    if not np.any((pose < 0) | (pose > size_px)) and self.map[pose[1]][pose[0]] == 255:
      valid = True

    return valid

  # Move in a random direction
  def move(self):
    # set standard deviations and means to sample from gaussian
    mu = 0
    sigma = 0.3

    # Update particle position
    self._pose[0] += sigma**2 * np.random.randn() + mu
    self._pose[1] += sigma**2 * np.random.randn() + mu
    self._pose[2] += 0

    # For a small probability the particles can be repositioned any valid location in the arena

    # 1% chance

    if np.random.rand() < 0.001:
      self.sample_random_pose()

    pass

  # Computed the weight dependent on the current position of the particle
  def compute_weight(self, police, baddies_list):
    baddie_low_th = 0.1
    baddie_high_th = 1
    gain = 0.5

    undetectable = True

    for police_car in police:
      robot_point = np.asarray(police_car.pose[:2])
      if np.linalg.norm(robot_point - self._pose[:2]) < 3 and not obstructed(self._pose[:2], robot_point, self.map):
        undetectable = False
        break

    # If undetectable give uniform weight
    if undetectable:
      weight = 1#np.random.rand()
    elif len(baddies_list) == 0:
      weight = 0
    else:
      # weight it according to the position of potential baddies
      weight = 0
      for baddie in baddies_list:
        dist = np.linalg.norm(baddie - self._pose[:2])
        if dist < baddie_low_th:
          weight = max(100*gain/baddie_low_th, weight)
        elif dist < baddie_high_th:
          weight = max(gain/dist, weight)

    #weight = 1


    # Check if particle is outside of the area. If outside reduce weights massively
    if not self.is_valid(self._pose):
      weight = weight * 0.001

    # update the weights
    self._weight = weight

  @property
  def pose(self):
    return self._pose

  @property
  def weight(self):
    return self._weight


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


def update_particles(particles, num_particles, frame_id, police, baddies_list):

  total_weight = 0.
  # move particles and calculate weight
  for i, p in enumerate(particles):
    p.move()
    if len(baddies_list) > i:
      p.set_pose(baddies_list[i][0], baddies_list[i][1])

    p.compute_weight(police, baddies_list)
    total_weight += p.weight

  # Low variance re-sampling of particles.
  new_particles = []
  random_weight = np.random.rand() * total_weight / num_particles
  current_boundary = particles[0].weight
  j = 0
  for m in range(len(particles)):
    next_boundary = random_weight + m * total_weight / num_particles
    while next_boundary > current_boundary:
      j = j + 1;
      if j >= num_particles:
        j = num_particles - 1
      current_boundary = current_boundary + particles[j].weight
    new_particles.append(copy.deepcopy(particles[j]))
  particles = new_particles

  # Help setup the publish process for particles.
  particle_msg = PointCloud()
  particle_msg.header.seq = frame_id
  particle_msg.header.stamp = rospy.Time.now()
  particle_msg.header.frame_id = '/map'
  intensity_channel = ChannelFloat32()
  intensity_channel.name = 'intensity'
  particle_msg.channels.append(intensity_channel)
  for p in particles:
    pt = Point32()
    pt.x = p.pose[X]
    pt.y = p.pose[Y]
    pt.z = .05
    particle_msg.points.append(pt)
    intensity_channel.values.append(p.weight)

  return particles, particle_msg

def world_to_pixel(point):
  xy = np.true_divide(point, resolution_m)
  xy[1] *= -1
  xy += size_px / 2
  xy = xy.astype(np.int32)
  return xy


def obstructed(point1, point2, map_img):
  p1_px = world_to_pixel(point1)
  p2_px = world_to_pixel(point2)
  obstructed = False

  if abs(p1_px[0] - p2_px[0]) > abs(p1_px[1] - p2_px[1]):
    if p1_px[0] > p2_px[0]:
      start_pt = p2_px
      end_pt = p1_px
    else:
      start_pt = p1_px
      end_pt = p2_px

    grad = (end_pt[1] - start_pt[1]) / (end_pt[0] - start_pt[0])

    y = start_pt[1]
    x = start_pt[0]
    for i in range(end_pt[0] - start_pt[0]):
      x_px = int(x + i)
      y_px = int(y)
      if map_img[y_px][x_px] == 0:
        obstructed = True
        break
      y += grad

  else:
    if p1_px[1] > p2_px[1]:
      start_pt = p2_px
      end_pt = p1_px
    else:
      start_pt = p1_px
      end_pt = p2_px

    grad = (end_pt[0] - start_pt[0]) / (end_pt[1] - start_pt[1])

    y = start_pt[1]
    x = start_pt[0]
    for i in range(end_pt[1] - start_pt[1]):
      x_px = int(x)
      y_px = int(y + i)
      if map_img[y_px][x_px] == 0:
        obstructed = True
        break
      x += grad
  return obstructed
