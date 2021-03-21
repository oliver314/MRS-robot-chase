from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import time


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Drawing constants.
REFRESH_RATE = 1. / 15.


# To call adaptive solution, use key dt=100 when running the file

TOL = 0.002

def set_thetas(T):
  w = lambda t: np.cos(np.floor(t))
  thetas = np.zeros(T)
  thetas[0] = 0
  for t in range(1, T):
    thetas[t] = thetas[t-1] + w(t-1)
  return thetas

   
def step_euler(t, dt, current_pose, step_sizes, thetas):
  u = 0.25
  next_pose = current_pose.copy()
  
  theta = lambda t: thetas[np.floor(t)] + (t-np.floor(t))*np.cos(np.floor(t))
  x_dot = lambda t: u * np.cos(theta(t))
  y_dot = lambda t: u * np.sin(theta(t))

  A1x = current_pose[X] + dt/2 * x_dot(t) + dt/2 * x_dot(t+dt/2)
  A2x = current_pose[X] + dt * x_dot(t)
  errorx = np.abs(A1x-A2x)/dt

  A1y = current_pose[Y] + dt/2 * y_dot(t) + dt/2 * y_dot(t+dt/2)
  A2y = current_pose[Y] + dt * y_dot(t)
  errory = np.abs(A1y-A2y)/dt

  error = np.max(errorx, errory)
  if error < TOL:
    next_pose[X] = 2*A2x-A1x
    next_pose[Y] = 2*A2y-A1y
    next_pose[YAW] = theta(t)
    t = t + dt
    step_sizes.append(dt)

  dt = 0.9*TOL/error*dt
  #print(dt)
  dt = min(dt, 0.3)
  return t, dt, next_pose
  

def adaptive_step_size_euler(fig, ax, color):

  dt = 'adaptive'
  # Initial robot pose (x, y and theta).
  robot_pose = np.array([0., 0., 0.], dtype=np.float32)
  robot_drawer = RobotDrawer(ax, robot_pose, color=color, label='dt = '+str(dt) +' [s]')
  if args.animate:
      fig.canvas.draw()
      fig.canvas.flush_events()

  # Simulate for 10 seconds.
  last_time_drawn = 0.
  last_time_drawn_real = time.time()


  t = 0.0
  dt = 1.0
  step_sizes = []
  thetas = set_thetas(1000)
  for i in range(2000):
    # execute loop 1000 times. Will be less than 1000 executed timesteps but will go further than 100 fixed size steps
    t, dt, robot_pose = step_euler(t, dt, robot_pose, step_sizes, thetas)

    plt.title('time = %.3f [s] with dt = %.3f [s]' % (t + dt, dt))
    robot_drawer.update(robot_pose)

    # Do not draw too many frames.
    time_drawn = t
    if args.animate and (time_drawn - last_time_drawn > REFRESH_RATE):
	# Try to draw in real-time.
	time_drawn_real = time.time()
	delta_time_real = time_drawn_real - last_time_drawn_real
	if delta_time_real < REFRESH_RATE:
	  time.sleep(REFRESH_RATE - delta_time_real)
	last_time_drawn_real = time_drawn_real
	last_time_drawn = time_drawn
	fig.canvas.draw()
	fig.canvas.flush_events()
  robot_drawer.done()
  print("Number accepted step sizes " + str(len(step_sizes)))
  print("Average accepted step sizes " + str(np.mean(step_sizes)))
  print("Time " + str(t))
  return step_sizes


def euler(current_pose, t, dt):
  '''Euler's integration method to return the next pose of our robot.

  Parameters:
     t is the current time.
     dt is the time-step duration.
     current_pose[X] is the current x position.
     current_pose[Y] is the current y position.
     current_pose[YAW] is the current orientation of the robot.
  Returns:
     next_pose[X], next_pose[Y], next_pose[YAW]
  '''
  next_pose = current_pose.copy()
  u = 0.25
  w = np.cos(np.floor(t))

  x_dot = u * np.cos(current_pose[YAW])
  y_dot = u * np.sin(current_pose[YAW])
  theta_dot = w

  next_pose[X] = current_pose[X] + dt * x_dot
  next_pose[Y] = current_pose[Y] + dt * y_dot
  next_pose[YAW] = current_pose[YAW] + dt * theta_dot
  
  #thetas = set_thetas(100)
  #print("Compare")
  #print(thetas[np.floor(t)] + (t-np.floor(t))*np.cos(np.floor(t)))
  #print(next_pose[YAW])
  return next_pose


def rk4(current_pose, t, dt):
  '''classical Runge-Kutta to return the next pose of our robot.

  Parameters
  
  Parameters:
     t is the current time.
     dt is the time-step duration.
     current_pose[X] is the current x position.
     current_pose[Y] is the current y position.
     current_pose[YAW] is the current orientation of the robot.
  Returns:
     next_pose[X], next_pose[Y], next_pose[YAW]
  '''
  next_pose = current_pose.copy()
  u = 0.25

  thetas = set_thetas(1000)
  theta = lambda t: thetas[np.floor(t)] + (t-np.floor(t))*np.cos(np.floor(t))

  k_th = lambda t: np.cos(np.floor(t))
  k_x = lambda th: u * np.cos(th)
  k_y = lambda th: u * np.sin(th)

  theta_n = theta(t) #current_pose[YAW]
  theta_n_plus_half = theta(t+dt/2) #current_pose[YAW] + 1/6.0 * dt * (k_th(t) + 4 * k_th(t+dt/2) + k_th(t+dt))
  theta_n_plus_1 = theta(t+dt) #current_pose[YAW] + 1/6.0 * dt/2.0 * (k_th(t) + 4 * k_th(t+dt/4) + k_th(t+dt/2))
  # Pose at time t+dt
  next_pose[X] = current_pose[X] + 1/6.0 * dt * (k_x(theta_n) + 4 * k_x(theta_n_plus_half) + k_x(theta_n_plus_1))
  next_pose[Y] = current_pose[Y] + 1/6.0 * dt * (k_y(theta_n) + 4 * k_y(theta_n_plus_half) + k_y(theta_n_plus_1))
  next_pose[YAW] = theta_n_plus_1

  return next_pose


def main(args):
  print('Using method {}'.format(args.method))
  integration_method = globals()[args.method]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.ion()  # Interactive mode.
  plt.grid('on')
  plt.axis('equal')
  plt.xlim([-0.5, 2])
  plt.ylim([-0.75, 1.25])
  plt.show()
  colors = colors_from('jet', len(args.dt))

  # Show all dt.
  for color, dt in zip(colors, args.dt):
    print('Using dt = {}'.format(dt))
    if(dt==100):
	step_sizes = adaptive_step_size_euler(fig, ax, color)
    else: 
            
	    # Initial robot pose (x, y and theta).
	    robot_pose = np.array([0., 0., 0.], dtype=np.float32)
	    robot_drawer = RobotDrawer(ax, robot_pose, color=color, label='dt = '+str(dt) +' [s]')
	    if args.animate:
	      fig.canvas.draw()
	      fig.canvas.flush_events()

	    # Simulate for 10 seconds.
	    last_time_drawn = 0.
	    last_time_drawn_real = time.time()
	    for t in np.arange(0., 200., dt):
	      robot_pose = integration_method(robot_pose, t, dt)
	      
	      plt.title('time = %.3f [s] with dt = %.3f [s]' % (t + dt, dt))
	      robot_drawer.update(robot_pose)

	      # Do not draw too many frames.
	      time_drawn = t
	      if args.animate and (time_drawn - last_time_drawn > REFRESH_RATE):
		# Try to draw in real-time.
		time_drawn_real = time.time()
		delta_time_real = time_drawn_real - last_time_drawn_real
		if delta_time_real < REFRESH_RATE:
		  time.sleep(REFRESH_RATE - delta_time_real)
		last_time_drawn_real = time_drawn_real
		last_time_drawn = time_drawn
		fig.canvas.draw()
		fig.canvas.flush_events()
    	    robot_drawer.done()

  plt.ioff()
  plt.title('Trajectories')
  plt.legend(loc='lower right')

  
  plt.figure()
  #plt.plot(step_sizes[:400])
  plt.xlabel("Iteration")
  plt.ylabel("Accepted Step size")
  plt.title("")
  plt.show()


# Simple class to draw and animate a robot.
class RobotDrawer(object):

  def __init__(self, ax, pose, radius=.05, label=None, color='g'):
    self._pose = pose.copy()
    self._radius = radius
    self._history_x = [pose[X]]
    self._history_y = [pose[Y]]
    self._outside = ax.plot([], [], 'b', lw=2)[0]
    self._front = ax.plot([], [], 'b', lw=2)[0]
    self._path = ax.plot([], [], c=color, lw=2, label=label)[0]
    self.draw()

  def update(self, pose):
    self._pose = pose.copy()
    self._history_x.append(pose[X])
    self._history_y.append(pose[Y])
    self.draw()

  def draw(self):
    a = np.linspace(0., 2 * np.pi, 20)
    x = np.cos(a) * self._radius + self._pose[X]
    y = np.sin(a) * self._radius + self._pose[Y]
    self._outside.set_data(x, y)
    r = np.array([0., self._radius])
    x = np.cos(self._pose[YAW]) * r + self._pose[X]
    y = np.sin(self._pose[YAW]) * r + self._pose[Y]
    self._front.set_data(x, y)
    self._path.set_data(self._history_x, self._history_y)

  def done(self):
    self._outside.set_data([], [])
    self._front.set_data([], [])


def colors_from(cmap_name, ncolors):
    cm = plt.get_cmap(cmap_name)
    cm_norm = matplotlib.colors.Normalize(vmin=0, vmax=ncolors - 1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=cm_norm, cmap=cm)
    return [scalar_map.to_rgba(i) for i in range(ncolors)]


def positive_floats(string):
  values = tuple(float(v) for v in string.split(','))
  for v in values:
    if v <= 0.:
      raise argparse.ArgumentTypeError('{} is not strictly positive.'.format(v))
  return values


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--method', action='store', default='euler', help='Integration method.', choices=['euler', 'rk4'])
  parser.add_argument('--dt', type=positive_floats, action='store', default=(0.05,), help='Integration step.')
  parser.add_argument('--animate', action='store_true', default=False, help='Whether to animate.')
  args = parser.parse_args()
  main(args)
