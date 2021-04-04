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
import rospy

from utils import SimpleLaser, GroundtruthPose, feedback_linearized


MAX_VELOCITY_POLICE = 0.6
MAX_VELOCITY_BADDIES = 1
MAX_ANGULAR_VELOCITY = 0.8

# define laser angles
laser_range = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
lidar_range = np.linspace(-np.pi,np.pi,50,endpoint=False).tolist()
#lidar_range = [-np.pi/2]

class actor(object):
    def __init__(self, name):
      self.laser = SimpleLaser(name,laser_range)
      self.lidar = SimpleLaser(name,lidar_range)
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

    def lidar_scan(self):
      while not self.lidar.ready:
        time.sleep(0.1)
      return self.lidar.measurements

    def set_vel_holonomic(self, x, y):
      u, w = feedback_linearized(self.pose, [x, y], 0.3)
      self.set_vel(u, w)

    def lidar_xy(self):
      state = self.pose
      theta = lidar_range + state[2]
      r = np.asarray(self.lidar_scan())
      xy = np.transpose(np.array([r*np.cos(theta) + state[0], r*np.sin(theta) + state[1]]))
      return xy

    @property
    def pose(self):
      return self._pose()



class baddie(actor):
  def __init__(self, name):
    self._caught = False
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



class police_car(actor):

  def set_vel(self, u, w):
    vel_msg = Twist()
    vel_msg.linear.x = max(-MAX_VELOCITY_POLICE, min(u, MAX_VELOCITY_POLICE))
    vel_msg.angular.z = max(-MAX_ANGULAR_VELOCITY, min(w, MAX_ANGULAR_VELOCITY))
    self.publisher.publish(vel_msg)