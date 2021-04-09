from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import matplotlib.pyplot as plt


from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Point32

# For groundtruth information.
import rospy

# Map occupancy information
size_m = 10         # 10m x 10m size
resolution_m = 0.01 # 1cm resolution
size_px = int(size_m/resolution_m) # image is of size: size_px x size_px


###############
# CLASSES
###############

class Particle(object):
  """Represents a particle."""

  def __init__(self, xlim, ylim, map_img):
    self._pose = np.zeros(3, dtype=np.float32)
    self._weight = 1.
    self.xlim = xlim
    self.ylim = ylim
    self.map = map_img

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
  def compute_weight(self, police, baddie_loc):
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
    elif baddie_loc is None:
      weight = 0
    else:
      # weight it according to the position of potential baddies
      weight = 0
      dist = np.linalg.norm(baddie_loc - self._pose[:2])
      if dist < baddie_low_th:
        weight = max(100*gain/baddie_low_th, weight)
      elif dist < baddie_high_th:
        weight = max(gain/dist, weight)


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


###############
# Functions
###############

def update_particles(particles, num_particles, frame_id, police, baddie_loc):

  total_weight = 0.
  # move particles and calculate weight
  for i, p in enumerate(particles):
    p.move()
    p.compute_weight(police, baddie_loc)
    total_weight += p.weight

  # Low variance re-sampling of particles.
  new_particles = []
  random_weight = np.random.rand() * total_weight / num_particles
  current_boundary = particles[0].weight
  j = 0
  for m in range(len(particles)):
    next_boundary = random_weight + m * total_weight / num_particles
    while next_boundary > current_boundary:
      j = j + 1
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
  if baddie_loc is not None:
    particles[0].set_pose(baddie_loc[0], baddie_loc[1])

  for i, p in enumerate(particles):
    pt = Point32()
    pt.x = p.pose[0]
    pt.y = p.pose[1]
    pt.z = .05
    particle_msg.points.append(pt)
    intensity_channel.values.append(p.weight)

  # Calculate mean vector and covariance matrix
  mu = np.array([0.,0.])
  for p in particles:
    mu[0] += p.pose[0] / num_particles
    mu[1] += p.pose[1] / num_particles

  sigma = np.array([[0.,0.],[0.,0.]])
  for p in particles:
    sigma[0][0] += (p.pose[0] - mu[0]) ** 2 / num_particles
    sigma[0][1] += (p.pose[0] - mu[0]) * (p.pose[1] - mu[1]) / num_particles
    sigma[1][0] += (p.pose[0] - mu[0]) * (p.pose[1] - mu[1]) / num_particles
    sigma[1][1] += (p.pose[1] - mu[1]) ** 2 / num_particles

  return particles, particle_msg, mu, sigma

# Convert a single point from world coordinates to pixel coordinates
def world_to_pixel(point):
  xy = np.true_divide(point, resolution_m)
  xy[1] *= -1
  xy += size_px / 2
  xy = xy.astype(np.int32)
  return xy

# Returns true if a line between two points in space are obstructed by an obstacle
# the obstacle is based on the map image
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


# Create a list of points which represents baddies which the police can see on its sight
def baddies_list_LoS(police, baddies, map_img):
  baddies_identified = [None] * len(baddies)

  for i, baddie in enumerate(baddies):
    for police_car in police:
      if not obstructed(police_car.pose[:2], baddie._pose()[:2], map_img):
        baddies_identified[i] = baddie._pose()[:2]
        break

  return baddies_identified


# Create a list of points which represents baddies which the police can see via a Lidar
def baddies_list_lidar(police, map_img, live_plot = True):

  baddies_identified = []

  xy_raw_list = []
  for police_car in police:
    xy_raw_list.append(police_car.lidar_xy)

  xy_raw = np.vstack(tuple(xy_raw_list))

  xy = np.true_divide(xy_raw, resolution_m)
  xy[:, 1] *= -1
  xy += size_px/2
  xy = xy.astype(np.int32)

  robot_PC = []
  detected_points = []
  for i, coord in enumerate(xy):
    if not np.any((coord < 0)|(coord > size_px)):
      detected_points.append(xy_raw[i])
      if map_img[coord[1]][coord[0]] == 255:
        robot_PC.append(xy_raw[i])
  robot_PC = np.asarray(robot_PC)
  detected_points = np.asarray(detected_points)

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

  identified = np.asarray(identified)


  # Exclude any positions that corresponds to your fellow officers from the list
  for est in identified:
    is_police = False
    for police_car in police:
      if np.linalg.norm(police_car.pose[:2] - est) < 0.2:
        is_police = True
        break

    if not is_police:
      baddies_identified.append(est)

  if np.shape(detected_points)[0] > 0 and live_plot:
    plt.clf()
    plt.scatter(detected_points[:,0], detected_points[:,1])
    plt.draw()
    plt.pause(0.001)

  # print(np.asarray(baddies_identified))
  # print("\n\n")


  return baddies_identified


# Link between previous assignment to baddies and new measurements
def sort_baddies(temp_list, baddies_list):
  output_list = [None, None, None]

  # If no baddies were detected in the first place
  if len(temp_list) == 0:
    return output_list

  # If the previous baddies list do not contain any baddies
  if all(baddie is None for baddie in baddies_list):
    for i in range(len(temp_list)):
      output_list[i] = temp_list[i]
    return output_list

  # one or more baddies were detected
  # The previous list contains some information
  recorded = []
  for n in range(len(temp_list)):
    baddie = baddies_list[n]
    if baddie is not None:
      closest = [10000., None]
      for i, pos in enumerate(temp_list):
        dist = np.linalg.norm(baddie - pos)
        if dist < closest[0]:
          closest[0] = dist
          closest[1] = i
      output_list[n] = temp_list[closest[1]]
      recorded.append(closest[1])

  # When there are more items in temp_list compared to baddies_list
  for i, pos in enumerate(temp_list):
    # if this particular position has not been recorded in the output list
    if all(index != i for index in recorded):
      for j, value in enumerate(output_list):
        if value is None:
          output_list[j] = pos
          recorded.append(i)
          break

  return output_list


def get_baddies_estimation(police, baddies, prev_baddie_measurement,line_of_sight, map_img, particles, particle_publisher, num_particles, idx, live_plot):

    if line_of_sight:
      curr_baddie_measurement = baddies_list_LoS(police, baddies, map_img)
    else:
      temp_list = baddies_list_lidar(police, map_img, live_plot=live_plot)
      curr_baddie_measurement = sort_baddies(temp_list, prev_baddie_measurement)

    # Move, compute weight, resample particles
    for i, baddie_loc in enumerate(curr_baddie_measurement):
      particles[i], particle_msg, mu, sigma = update_particles(particles[i], num_particles, idx, police, baddie_loc)

      # Set the estimated position and variance of each baddie
      baddies[i].set_mu_sigma(mu, sigma)

      # publish particles
      particle_publisher[i].publish(particle_msg)

    return curr_baddie_measurement