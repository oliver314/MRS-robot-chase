from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import time
from enum import Enum


class RobotType(Enum):
    BADDIE = 1
    POLICE = 2


class Robot(object):
    def __init__(self, ax, pose, robot_type, draw=True, radius=.1, max_velocity=1, label=None):
        self.do_drawing = draw
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_velocity
        self.caught = False
        self.catch_radius = 0.35
        self._show_catch_radius = True
        self._show_trail = False
        self._show_lidar = False
        self._color = "b" if robot_type == RobotType.POLICE else "r"
        self.velocity = [0, 0]
        self.pose = pose.copy()
        self._radius = radius
        self._history_x = [pose[X]]
        self._history_y = [pose[Y]]

        self._lidar_angles = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
        self.last_lidar_readings = [1] * len(self._lidar_angles)
        self.robot_type = robot_type
        self._lidar_lines = []
        if draw:
            self.text = ax.text(0, 0, "", ha="center", va="center")
            self._catch_circle = ax.plot([], [], "grey", lw=1)[0]
            self._outside = ax.plot([], [], self._color, lw=2)[0]
            self._front = ax.plot([], [], self._color, lw=2)[0]

            if self._show_lidar:
                for i in range(len(self._lidar_angles)):
                    self._lidar_lines.append(ax.plot([], [])[0])

            if self._show_trail:
                self._path = ax.plot([], [], c=self._color, lw=2, label=label)[0]

            self.draw()

    def update(self, pose):
        self.pose = pose.copy()
        self._history_x.append(pose[X])
        self._history_y.append(pose[Y])

        self.update_lidar()

        if self.do_drawing:
            self.draw()

    def draw(self):
        a = np.linspace(0., 2 * np.pi, 20)
        x = np.cos(a) * self._radius + self.pose[X]
        y = np.sin(a) * self._radius + self.pose[Y]
        self._outside.set_data(x, y)
        r = np.array([0., self._radius])
        x = np.cos(self.pose[YAW]) * r + self.pose[X]
        y = np.sin(self.pose[YAW]) * r + self.pose[Y]
        self._front.set_data(x, y)

        if self._show_trail:
            self._path.set_data(self._history_x, self._history_y)

        if self._show_catch_radius and self.robot_type == RobotType.POLICE:
            x = np.cos(a) * self.catch_radius + self.pose[X]
            y = np.sin(a) * self.catch_radius + self.pose[Y]
            self._catch_circle.set_data(x, y)

        if self._show_lidar:
            self.draw_lidar()

    def update_lidar(self):
        for n, angle in enumerate(self._lidar_angles):
            dist = self.ray_trace(angle)

            self.last_lidar_readings[n] = dist

    def draw_lidar(self):
        for n, angle in enumerate(self._lidar_angles):
            x = self.pose[0] + np.cos(self.pose[2] + angle) * self.last_lidar_readings[n]
            y = self.pose[1] + np.sin(self.pose[2] + angle) * self.last_lidar_readings[n]

            self._lidar_lines[n].set_data([self.pose[0], x], [self.pose[1], y])

    def done(self):
        self._outside.set_data([], [])
        self._front.set_data([], [])

    def catch(self):
        self.caught = True

        if self.do_drawing:
            self.text.set_x(self.pose[0])
            self.text.set_y(self.pose[1] + self._radius + 0.3)
            self.text.set_text("Caught")

    def set_vel(self, u, w):
        u = min(self.max_velocity, u)
        w = min(self.max_angular_velocity, w)

        self.velocity = [u, w]

    def set_vel_holonomic(self, x, y):
        u, w = feedback_linearized(self.pose, [x, y], 0.2)

        self.set_vel(u, w)

    def ray_trace(self, angle):
        """Returns the distance to the first obstacle from the particle."""
        def intersection_segment(x1, x2, y1, y2):
            point1 = np.array([x1, y1], dtype=np.float32)
            point2 = np.array([x2, y2], dtype=np.float32)
            v1 = self.pose[:2] - point1
            v2 = point2 - point1
            v3 = np.array([np.cos(angle + self.pose[YAW] + np.pi / 2.), np.sin(angle + self.pose[YAW] + np.pi / 2.)],
                          dtype=np.float32)
            t1 = np.cross(v2, v1) / np.dot(v2, v3)
            t2 = np.dot(v1, v3) / np.dot(v2, v3)
            # t1[np.logical_or(np.logical_or(t1 < 0, t2 < 0), t2 > 1)] = float('inf')
            # return t1 => error in intersection_cyliner. Not going any further into this
            if t1 >= 0. and t2 >= 0. and t2 <= 1.:
                return t1
            return float('inf')

        def intersection_cylinder(x, y, r):
            center = np.array([x, y], dtype=np.float32)
            v = np.array([np.cos(angle + self.pose[YAW] + np.pi), np.sin(angle + self.pose[YAW] + np.pi)],
                         dtype=np.float32)

            v1 = center - self.pose[:2]
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

        lowest = float('inf')
        for line in lines:
            lowest = min(intersection_segment(line[0][0], line[1][0], line[0][1], line[1][1]), lowest)

        for i in range(len(CYLINDER_POSITIONS)):
            lowest = min(intersection_cylinder(CYLINDER_POSITIONS[i][0], CYLINDER_POSITIONS[i][1], CYLINDER_RADIUSS[i]), lowest)

        return lowest


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Drawing constants.
REFRESH_RATE = 1. / 15.

robot_config = [
    [np.array([-3., 0., 0.], dtype=np.float32), RobotType.POLICE],
    [np.array([-3., 1, 0.], dtype=np.float32), RobotType.POLICE],
    [np.array([-3., -1, 0.], dtype=np.float32), RobotType.POLICE],
    [np.array([3., 0., 0.], dtype=np.float32), RobotType.BADDIE],
    [np.array([3., 1, 0.], dtype=np.float32), RobotType.BADDIE],
    [np.array([3., -1, 0.], dtype=np.float32), RobotType.BADDIE]
]

CYLINDER_POSITIONS = np.array([
    [0, -1],
    [0, 1]
])

CYLINDER_RADIUSS = np.array([
    0.5,
    0.25
])

WALL_OFFSET = 4

lines = [
    ((-4, -4), (4, -4)),
    ((4, -4), (4, 4)),
    ((4, 4), (-4, 4)),
    ((-4, 4), (-4, -4))
]


def deal_with_collisions(robots, robot, v):
    for line in lines:
        a = np.array(line[0])
        b = np.array(line[1])

        p = np.array(robot.pose[:2])

        n = b - a
        n = n / np.linalg.norm(n)

        robot_to_line = (a - p) - np.dot((a - p), n) * n

        if np.linalg.norm(robot_to_line) < robot._radius:
            component = robot_to_line * np.dot(v, robot_to_line) / np.dot(robot_to_line, robot_to_line)

            if np.dot(component, robot_to_line) > 0:
                v = v - component

    for i in range(len(CYLINDER_POSITIONS)):
        center = CYLINDER_POSITIONS[i]
        r = CYLINDER_RADIUSS[i]

        if np.linalg.norm(center - robot.pose[:2]) < robot._radius + r:
            robot_to_center = center - robot.pose[:2]

            component = robot_to_center * np.dot(v, robot_to_center) / np.dot(robot_to_center, robot_to_center)

            if np.dot(component, robot_to_center) > 0:
                v = v - component

    for other_robot in robots:
        if other_robot == robot:
            continue

        center = np.array(other_robot.pose[:2])
        r = other_robot._radius

        if np.linalg.norm(center - robot.pose[:2]) < robot._radius + r:
            robot_to_center = center - robot.pose[:2]

            component = robot_to_center * np.dot(v, robot_to_center) / np.dot(robot_to_center, robot_to_center)

            if np.dot(component, robot_to_center) > 0:
                v = v - component

    return v


def is_in_valid_position(robot, robots):
    for line in lines:
        a = np.array(line[0])
        b = np.array(line[1])

        p = np.array(robot.pose[:2])

        n = b - a
        n = n / np.linalg.norm(n)

        robot_to_line = (a - p) - np.dot((a - p), n) * n

        if np.linalg.norm(robot_to_line) < robot._radius:
            return False

    for i in range(len(CYLINDER_POSITIONS)):
        center = CYLINDER_POSITIONS[i]
        r = CYLINDER_RADIUSS[i]

        if np.linalg.norm(center - robot.pose[:2]) < robot._radius + r:
            return False

    for other_robot in robots:
        if other_robot == robot:
            continue

        center = np.array(other_robot.pose[:2])
        r = other_robot._radius

        if np.linalg.norm(center - robot.pose[:2]) < robot._radius + r:
            return False

    return True


# def keyb_event(event, v):
#     if event.key == "left":
#         robots[0].velocity[1] = v * 2
#     if event.key == "right":
#         robots[0].velocity[1] = -v * 2
#     if event.key == "up":
#         robots[0].velocity[0] = v
#     if event.key == "down":
#         robots[0].velocity[0] = -v
#
#     if event.key == "a":
#         robots[3].velocity[1] = v * 2
#     if event.key == "d":
#         robots[3].velocity[1] = -v * 2
#     if event.key == "w":
#         robots[3].velocity[0] = v
#     if event.key == "s":
#         robots[3].velocity[0] = -v


def check_caught(robots):
    for i, robot in enumerate(robots):
        for j, robot2 in enumerate(robots):
            if robot.robot_type == RobotType.BADDIE and robot2.robot_type == RobotType.POLICE:
                dist = np.linalg.norm(robot.pose[:2] - robot2.pose[:2])
                if dist < robot.catch_radius:
                    robot.catch()
                    break


def all_baddies_caught(robots):
    for robot in robots:
        if not robot.caught and robot.robot_type == RobotType.BADDIE:
            return False

    return True


def euler(t, dt, robot, robots):
    nextpose = robot.pose.copy()
    u = robot.velocity[0]
    w = robot.velocity[1]

    v = np.array([0., 0.])

    v[0] = u * np.cos(robot.pose[YAW])
    v[1] = u * np.sin(robot.pose[YAW])
    theta_dot = w

    v = deal_with_collisions(robots, robot, v)

    nextpose[X] = robot.pose[X] + dt * v[0]
    nextpose[Y] = robot.pose[Y] + dt * v[1]
    nextpose[YAW] = robot.pose[YAW] + dt * theta_dot
    return nextpose


def blank(robots):
    pass


def run_simulation(headless=False, max_time=100, velocity_update_function=blank):
    #global robots
    integration_method = euler

    if not headless:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.mpl_connect('key_press_event', lambda e: keyb_event(e, .4))
        fig.canvas.mpl_connect("key_release_event", lambda e: keyb_event(e, 0))
        plt.ion()  # Interactive mode.
        plt.grid('on')
        plt.axis('equal')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

        for line in lines:
            plt.plot((line[0][0], line[1][0]), (line[0][1], line[1][1]))

        for i in range(len(CYLINDER_POSITIONS)):
            ax.add_patch(plt.Circle(CYLINDER_POSITIONS[i], CYLINDER_RADIUSS[i]))


        plt.show()

    robots = []

    dt = 0.05

    # Show all dt.
    for i, config in enumerate(robot_config):
        robots.append(Robot(ax if not headless else None, config[0], config[1], label='dt = %.3f [s]' % dt, draw=not headless, max_velocity=0.5 if config[1] == RobotType.BADDIE else 1))

        while True:
            x = np.random.uniform(low=-WALL_OFFSET, high=WALL_OFFSET)
            y = np.random.uniform(low=-WALL_OFFSET, high=WALL_OFFSET)
            theta = np.random.uniform(low=-np.pi, high=np.pi)

            robots[-1].pose = np.array([x, y, theta])

            if is_in_valid_position(robots[-1], robots):
                break

        if not headless:
            fig.canvas.draw()
            fig.canvas.flush_events()

    catch_times = [max_time] * 3

    # Simulate for 10 seconds.
    last_time_drawn = 0.
    last_time_drawn_real = time.time()
    t = 0
    while not all_baddies_caught(robots) and t < max_time:
        velocity_update_function(robots)
        for n, robot in enumerate(robots):
            if not robot.caught:
                robot.update(integration_method(t, dt, robot, robots))

            if robot.robot_type == RobotType.BADDIE and robot.caught and t < catch_times[n-3]:
                catch_times[n-3] = t

        check_caught(robots)

        # Do not draw too many frames.
        time_drawn = t
        if (time_drawn - last_time_drawn > REFRESH_RATE) and not headless:
            plt.title('time = %.3f [s] with dt = %.3f [s]' % (t + dt, dt))
            # Try to draw in real-time.
            time_drawn_real = time.time()
            delta_time_real = time_drawn_real - last_time_drawn_real
            if delta_time_real < REFRESH_RATE:
                time.sleep(REFRESH_RATE - delta_time_real)
            last_time_drawn_real = time_drawn_real
            last_time_drawn = time_drawn
            fig.canvas.draw()
            fig.canvas.flush_events()

        t += dt

    for n, robot in enumerate(robots):
        if robot.robot_type == RobotType.BADDIE and robot.caught and t < catch_times[n - 3]:
            catch_times[n - 3] = t

    print(catch_times)

    score = 0

    for ctime in catch_times:
        score += max_time - ctime
        if ctime >= max_time:
            score -= max_time/2

    return score


def positive_floats(string):
    values = tuple(float(v) for v in string.split(','))
    for v in values:
        if v <= 0.:
            raise argparse.ArgumentTypeError('{} is not strictly positive.'.format(v))
    return values


def feedback_linearized(pose, velocity, epsilon):
    '''feedback-linearization to follow the velocity
    vector given as argument. Epsilon corresponds to the distance of
    linearized point in front of the robot.
    '''
    u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW]) #m/s
    w = (1/epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW])) # [rad/s] going counter-clockwise.
    return u, w


if __name__ == "__main__":
    print(run_simulation(False))
