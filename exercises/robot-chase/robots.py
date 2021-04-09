#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import math

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import rospy

from utils import SimpleLaser, GroundtruthPose, feedback_linearized


MAX_VELOCITY_POLICE = 0.6
MAX_VELOCITY_BADDIES = 1
MAX_ANGULAR_VELOCITY = 0.8

# define laser angles
laser_range = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]

class actor(object):
    def __init__(self, name):
      self.laser = SimpleLaser(name,laser_range)
      self.groundtruth = GroundtruthPose(name)
      self.publisher = rospy.Publisher("/%s/cmd_vel" % name, Twist, queue_size=5)
      self.name = name

    def _pose(self):
      while not self.groundtruth.ready:
        time.sleep(0.001)
      return self.groundtruth.pose

    def scan(self):
      while not self.laser.ready:
        time.sleep(0.1)
      return self.laser.measurements


    def set_vel_holonomic(self, x, y, backup=False):

      u, w = feedback_linearized(self.pose, [x, y], 0.2)
      
      if backup:
        # could also have a stuck_history variable, counting how long the police car hasn t moved for
        front, front_right, front_left, _, _ = self.scan()
        #print(front)
        if front <0.22 or front_left < 0.2 or front_right < 0.2:
          print("Emergency rule based controller jumps in with front = " + str(front))
          u = -0.2
          if front_left < front_right:
            w = -5
          else:
            w = 5

      self.set_vel(u, w)

    @property
    def pose(self):
        return self._pose()

    @property
    def gt_pose(self):
        return self._pose()


class baddie(actor):
  def __init__(self, name, access_to_gt_pose=True):
    self._caught = False
    self.est_pose = np.array([100] * 2)
    self.est_variance = np.array([[100] * 2,[100] * 2])
    self.access_to_gt_pose = access_to_gt_pose
    self.temp_caught_flag = False
    super(baddie, self).__init__(name)

  def set_vel(self, u, w):
    vel_msg = Twist()
    if self.caught:
      vel_msg.linear.x = 0
      vel_msg.angular.z = 0
    else:
      vel_msg.linear.x = max(-MAX_VELOCITY_BADDIES, min(u, MAX_VELOCITY_BADDIES))
      vel_msg.angular.z = max(-MAX_ANGULAR_VELOCITY, min(w, MAX_ANGULAR_VELOCITY))
    self.publisher.publish(vel_msg)

  @property
  def caught(self):
    return self._caught

  @caught.setter
  def caught(self, value):
    if not self.caught:
      print("%s caught!" % self.name)
    self._caught = value


  def set_mu_sigma(self, mu, sigma):
    self.est_pose = mu
    self.est_variance = sigma


  @property
  def pose(self):
      if self.access_to_gt_pose:
        return self._pose()
      else:
        return np.array([self.est_pose[0], self.est_pose[1], 0])


class police_car(actor):
  def __init__(self, name):
      self.xy = np.transpose(np.asarray([[float('inf')] * 360,[float('inf')] * 360]))
      rospy.Subscriber('/%s/scan' % name, LaserScan, self.lidar_callback)
      super(police_car, self).__init__(name)

  def set_vel(self, u, w):
    vel_msg = Twist()
    vel_msg.linear.x = max(-MAX_VELOCITY_POLICE, min(u, MAX_VELOCITY_POLICE))
    vel_msg.angular.z = max(-MAX_ANGULAR_VELOCITY, min(w, MAX_ANGULAR_VELOCITY))
    self.publisher.publish(vel_msg)


  def lidar_scan(self):
    while not self.lidar.ready:
      time.sleep(0.1)
    return self.lidar.measurements

  
  def lidar_callback(self, msg):
      theta = np.linspace(0, 2*np.pi, 360, endpoint=False) + self.pose[2]
      r = msg.ranges
      self.xy = np.transpose(np.array([r*np.cos(theta) + self.pose[0], r*np.sin(theta) + self.pose[1]]))


  @property
  def lidar_xy(self):
    return self.xy

