#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import numpy as np
import rospy
import time

from scipy.stats import norm

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
# For displaying particles.
# http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
# Odometry.
from nav_msgs.msg import Odometry


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

ROBOT_RADIUS = 0.105 / 2.
WALL_OFFSET = 2.
CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)
CYLINDER_RADIUS = .3 + ROBOT_RADIUS

PL = True

def braitenberg(front, front_left, front_right, left, right):
  '''a braitenberg controller that takes the range
 
  measurements given in argument to steer the robot safely
  '''
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # take inverse: make a very small input stand out, do not care about large inputs
  input_data = np.transpose([1/left, 1/front_left, 1/front, 1/front_right, 1/right, 1])

  #print(input_data)
  
  w1 = np.array([ 0, -0.4, -0.7, -0.4,  0, 1.8], dtype=float)
  w2 = np.array([-1,   -1,  0,    1,  1, 0], dtype=float)

  u = np.tanh(w1.dot(input_data))
  w = 0.6 * np.tanh(w2.dot(input_data))

  return u, w


class Particle(object):
  """Represents a particle."""

  def __init__(self):
    '''Initialize a valid particle randomly in the arena'''
    self._weight = 1.

    self._pose = np.random.uniform(-2, 2, 3)
    while(not self.is_valid()):
      self._pose = np.random.uniform(-2, 2, 3)
    self._pose[YAW] = np.random.uniform(-np.pi, np.pi)

  def is_valid(self):
    '''return True if the current particle position is valid. May be used in init and compute_weight'''
    
    if abs(self._pose[X]) > WALL_OFFSET or abs(self._pose[Y]) > WALL_OFFSET:
        # outside of box
        return False
    if (self._pose[X] - CYLINDER_POSITION[X])**2 + (self._pose[Y] -  CYLINDER_POSITION[Y])**2 < CYLINDER_RADIUS**2:
        # inside cylinder
	return False
    return True


  def move(self, delta_pose):
    '''Update the particle pose according the motion model.
    delta_pose is an offset in the particle frame. As motion model,
    use roughtly 10% standard deviation with respect to the forward
    and rotational velocity.
    '''
    # delta_pose evaluated as (sum in between two calls of) u*dt for x, 0 for y, w*dt for YAW 
    # well this only works if we call delta_pose more often than it is updated,
    # otherwise if it has rotated by 90 deg and moved forwards by one meter we won't know if 1m to left or forwards
    
    # noisy estimate on delta pose
    rel_std = 0.2
    noisy_delta_pose_sample = norm.rvs(delta_pose, rel_std*abs(delta_pose), 3)
    # sample will still have Y = 0 as desired (noisy estimate only on velocities)

    # generate rotation matrix from robot coordinates to world coordinates
    th = self._pose[YAW] + noisy_delta_pose_sample[YAW]/2
    T = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.]])
    delta_pose_sample_world = T.dot(noisy_delta_pose_sample)

    # update position in world coordinates
    self._pose += delta_pose_sample_world

    # with a low probability, the particle can be repositioned randomly in the arena
    # corresponds to probability of robot being kidnapped: one particle per time step
    if np.random.uniform() < 0.02:
        self.__init__()

  def compute_weight(self, front, front_left, front_right, left, right):
    '''Update the particle weight self._weight according to measurements.

    You can use the self.ray_trace(angle) function below 
    As measurement model, use a Gaussian error with a standard deviation of 80 [cm]. 
    We assume measurements are uncorrelated.
    Note that the maximum range of the laser-range finder is 3.5 meters 
    (observations beyond this range will show up as infinity).
    '''
    sigma = 0.8
    variance = sigma ** 2.
    invCovMatrix = 1/variance * np.eye(5)

    measurements = np.array([left, front_left, front, front_right, right])
    measurements = np.minimum(3.5, measurements)
    angles = np.array([np.pi / 2., np.pi / 4., 0., -np.pi / 4., -np.pi / 2.])

    # because ray_trace is not np friendly we have to do this in a loop
    pred_measurements = np.zeros(5)
    for i, angle in enumerate(angles):
            pred_measurements[i] = self.ray_trace(angle)

    error = measurements - pred_measurements
    if (measurements==3.5).all():
        # special case for erronous measurements first few steps
        self._weight = 1
    else:
        # 3 methods were explored in the hope to find one which converges faster. It is beyond the scope
        # of this exercise to quantitatively assess which one is best.  
        # First idea
        # Multivariate Gaussian is used, with diagonal covariance matrix, 
        # so it can be extended if we wanted to use correlation between measurements
        # we don't need the normalising constants because will need to be normalised anyway
        self._weight = np.exp(-0.5 * error.dot(invCovMatrix).dot(error))
 
    	# Second idea
        # An even more accurate model would be to use a truncated Gaussian
        # p(z=3.5 given x) = integral from 3.5 to inf of N(x, variance). Possibly limt second term
        # but this does not work very well. Possible because Gaussian approximation breaks down
	'''
        prob = 1
        for measurement, pred_measurement in zip(measurements, pred_measurements):
	   d = norm(loc = pred_measurement, scale = sigma)
	   if measurement == 3.5:
	      prob *= (d.cdf(np.inf)-d.cdf(3.5))
	   else:
	      prob *= d.pdf(measurement)
        self._weight = prob
	'''
	# Third idea
        # What if we pursue the idea of a truncated Gaussian but at least scale by what is too much
        # because that differs depending on the position of x
	'''	
	prob = 1
        for measurement, pred_measurement in zip(measurements, pred_measurements):
	   d = norm(loc = pred_measurement, scale = sigma)
           area = d.cdf(3.5) - d.cdf(0)
	   prob *= d.pdf(measurement)/area
        self._weight = prob
	'''

    if not self.is_valid():
	# reduce the weight of particles that are outside the arena.
	self._weight /= 10.

        

  def ray_trace(self, angle):
    """Returns the distance to the first obstacle from the particle."""
    def intersection_segment(x1, x2, y1, y2):
      point1 = np.array([x1, y1], dtype=np.float32)
      point2 = np.array([x2, y2], dtype=np.float32)
      v1 = self._pose[:2] - point1
      v2 = point2 - point1
      v3 = np.array([np.cos(angle + self._pose[YAW] + np.pi / 2.), np.sin(angle + self._pose[YAW]  + np.pi / 2.)],
                    dtype=np.float32)
      t1 = np.cross(v2, v1) / np.dot(v2, v3)
      t2 = np.dot(v1, v3) / np.dot(v2, v3)
      #t1[np.logical_or(np.logical_or(t1 < 0, t2 < 0), t2 > 1)] = float('inf')
      #return t1 => error in intersection_cyliner. Not going any further into this
      if t1 >= 0. and t2 >= 0. and t2 <= 1.:
        return t1
      return float('inf')

    def intersection_cylinder(x, y, r):
      center = np.array([x, y], dtype=np.float32)
      v = np.array([np.cos(angle + self._pose[YAW] + np.pi), np.sin(angle + self._pose[YAW] + np.pi)],
                   dtype=np.float32)
      
      v1 = center - self._pose[:2]
      a = v.dot(v)
      b = 2. * v.dot(v1)
      c = v1.dot(v1) - r ** 2.
      q = b ** 2. - 4. * a * c
      if q < 0.:
        return float('inf')
      g = 1. / (2. * a)
      q = g * np.sqrt(q)
      b = -b * g
      d = min(b + q, b - q)
      if d >= 0.:
        return d
      return float('inf')
    d = min(intersection_segment(-WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
            intersection_segment(WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, WALL_OFFSET),
            intersection_segment(-WALL_OFFSET, WALL_OFFSET, -WALL_OFFSET, -WALL_OFFSET),
            intersection_segment(-WALL_OFFSET, WALL_OFFSET, WALL_OFFSET, WALL_OFFSET),
            intersection_cylinder(CYLINDER_POSITION[X], CYLINDER_POSITION[Y], CYLINDER_RADIUS))
    return d

  @property
  def pose(self):
    return self._pose

  @property
  def weight(self):
    return self._weight


class SimpleLaser(object):
  def __init__(self):
    if PL:
      rospy.Subscriber('/police1/scan', LaserScan, self.callback)
    else:
      rospy.Subscriber('/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 3.1  # 3.1 degrees cone of view (3 rays).
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


class Motion(object):
  def __init__(self):
    self._previous_time = None
    self._delta_pose = np.array([0., 0., 0.], dtype=np.float32)
    if PL:
      rospy.Subscriber('/police1/odom', Odometry, self.callback)
    else:
      rospy.Subscriber('/odom', Odometry, self.callback)

  def callback(self, msg):
    u = msg.twist.twist.linear.x
    w = msg.twist.twist.angular.z
    if self._previous_time is None:
      self._previous_time = msg.header.stamp
    current_time = msg.header.stamp
    dt = (current_time - self._previous_time).to_sec()
    self._delta_pose[X] += u * dt
    self._delta_pose[Y] += 0.
    self._delta_pose[YAW] += w * dt
    self._previous_time = current_time
    #print("Delta pose updated")

  @property
  def ready(self):
    return True

  @property
  def delta_pose(self):
    #print("Delta pose consumed")
    ret = self._delta_pose.copy()
    self._delta_pose[:] = 0
    return ret


class GroundtruthPose(object):
  def __init__(self, name='turtlebot3_burger'):
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._name = name

  def callback(self, msg):
    idx = [i for i, n in enumerate(msg.name) if n == self._name]
    if not idx:
      raise ValueError('Specified name "{}" does not exist.'.format(self._name))
    idx = idx[0]
    self._pose[X] = msg.pose[idx].position.x
    self._pose[Y] = msg.pose[idx].position.y
    _, _, yaw = euler_from_quaternion([
        msg.pose[idx].orientation.x,
        msg.pose[idx].orientation.y,
        msg.pose[idx].orientation.z,
        msg.pose[idx].orientation.w])
    self._pose[YAW] = yaw

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose
  

def run(args):
  rospy.init_node('localization')

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  if PL:
    publisher = rospy.Publisher('/police1/cmd_vel', Twist, queue_size=5)
  else:
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  particle_publisher = rospy.Publisher('/particles', PointCloud, queue_size=1)
  laser = SimpleLaser()
  motion = Motion()
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = GroundtruthPose()
  pose_history = []
  with open('/tmp/gazebo_exercise.txt', 'w'):
    pass

  num_particles = 50
  particles = [Particle() for _ in range(num_particles)]

  frame_id = 0
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not laser.ready or not motion.ready or not groundtruth.ready:
      rate_limiter.sleep()
      continue

    u, w = braitenberg(*laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Update particle positions and weights.
    total_weight = 0.
    delta_pose = motion.delta_pose
    for i, p in enumerate(particles):
      p.move(delta_pose)
      p.compute_weight(*laser.measurements)
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

    # Publish particles.
    particle_msg = PointCloud()
    particle_msg.header.seq = frame_id
    particle_msg.header.stamp = rospy.Time.now()
    if PL:
      particle_msg.header.frame_id = '/police1/odom'
    else:
      particle_msg.header.frame_id = '/odom'
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
    particle_publisher.publish(particle_msg)

    # Log groundtruth and estimated positions in /tmp/gazebo_exercise.txt
    poses = np.array([p.pose for p in particles], dtype=np.float32)
    median_pose = np.median(poses, axis=0)
    pose_history.append(np.concatenate([groundtruth.pose, median_pose], axis=0))
    if len(pose_history) % 10:
      with open('/tmp/gazebo_exercise.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []
    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs a particle filter')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
