from simulator import *
import numpy as np
import neat
import visualize


def braitenberg_controller(lidar_readings):
    left, front_left, front, front_right, right = reversed(lidar_readings)
    u = 1.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.
    # take inverse: make a very small input stand out, do not care about large inputs
    input_data = np.transpose([1 / left, 1 / front_left, 1 / front, 1 / front_right, 1 / right, 1])
    w1 = np.array([0, -0.5, -1, -0.5, 0, 3], dtype=float)
    w2 = np.array([-1, -1, 0, 1, 1, 0], dtype=float)
    u = np.tanh(w1.dot(input_data))
    w = 0.6 * np.tanh(w2.dot(input_data))
    return u, w


def police_closest_method(police, baddies):
    for police_car in police:
        closest_baddie = None
        min_distance = np.inf
        for baddie in baddies:
            if np.linalg.norm(police_car.pose[:2] - baddie.pose[:2]) < min_distance and not baddie.caught:
                min_distance = np.linalg.norm(police_car.pose[:2] - baddie.pose[:2])
                closest_baddie = baddie
        police_car.set_vel_holonomic(*(-police_car.pose[:2] + closest_baddie.pose[:2]))


def baddies_pot_field_method(baddies, police):
    '''potential field method for baddies: assumes baddies know location of police'''
    P_gain_repulsive = 2.5
    rep_cutoff_distance = 2
    for baddie in baddies:
        # Have police cars as obstacles
        obstacle_positions = np.append(CYLINDER_POSITIONS, np.array([police_car.pose[:2] for police_car in police]), 0)
        obstacle_radii = np.append(CYLINDER_RADIUSS, [0.01 for police_car in police], 0)

        v = get_repulsive_field_from_obstacles(baddie.pose[:2], P_gain_repulsive, rep_cutoff_distance, WALL_OFFSET,
                                               obstacle_positions, obstacle_radii)
        baddie.set_vel_holonomic(*v)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-2:
        return np.zeros_like(v)
    return v / n


def get_repulsive_field_from_obstacles(position, a, b, WALL_OFFSET, obstacle_positions, obstacle_radii):
    v = np.zeros(2, dtype=np.float32)

    def adapt_v(delta_x_vector, v):
        distance = np.linalg.norm(delta_x_vector)
        v_cand = - a * (1. / b - 1 / distance) * 1 / distance ** 2 * delta_x_vector
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


def velocity_function_police_closest_baddie_braitenburg(robots):
    police = [robot for robot in robots if robot.robot_type == RobotType.POLICE]
    baddies = [robot for robot in robots if robot.robot_type == RobotType.BADDIE]

    for i in range(len(robots)):
        if robots[i].robot_type == RobotType.BADDIE:
            u, w = braitenberg_controller(robots[i].last_lidar_readings)

            robots[i].set_vel(u, w)
        else:
            police_closest_method(police, baddies)


def velocity_function_all_braitenberg(robots):
    for i in range(len(robots)):
        u, w = braitenberg_controller(robots[i].last_lidar_readings)

        robots[i].velocity = [u, w]


def velocity_function_police_closest_baddie_potential(robots):
    police = [robot for robot in robots if robot.robot_type == RobotType.POLICE]
    baddies = [robot for robot in robots if robot.robot_type == RobotType.BADDIE]

    for i in range(len(robots)):
        if robots[i].robot_type == RobotType.BADDIE:
            baddies_pot_field_method(baddies, police)
        else:
            police_closest_method(police, baddies)


def create_velocity_function(net):
    def velocity_function(robots):
        police = [robot for robot in robots if robot.robot_type == RobotType.POLICE]
        baddies = [robot for robot in robots if robot.robot_type == RobotType.BADDIE]

        #inputs = [*([*robot.pose, *robot.last_lidar_readings[1:4]] for robot in police), *[robot.pose[:2] for robot in baddies]]
        inputs = []

        for robot in police:
            inputs += [*list(robot.pose), *list(robot.last_lidar_readings[1:4])]

        for robot in baddies:
            inputs += [*list(robot.pose[:2])]

        outputs = net.activate(inputs)

        for i in range(3):
            police[i].set_vel(outputs[i * 2], outputs[i * 2 + 1])

        baddies_pot_field_method(baddies, police)

    return velocity_function


class LiveVisualReporter(neat.reporting.BaseReporter):
    def end_generation(self, config, population, species_set):
        global gen_no
        #visualize.plot_stats(stats, ylog=False, view=False, filename="gen" + str(gen_no) + "-stats-" + datetime.now().strftime("%d-%m-%Y---%H:%M:%S"))
        #visualize.plot_species(stats, view=False, filename="gen" + str(gen_no) + "-species-" + datetime.now().strftime("%d-%m-%Y---%H:%M:%S"))
        gen_no += 1


cpointer = neat.Checkpointer(1)

pop = cpointer.restore_checkpoint("neat-checkpoint-336")

genome = pop.population[19513]

config = pop.config

net = neat.nn.FeedForwardNetwork.create(genome, config)

vfunc = create_velocity_function(net)


run_simulation(headless=False, velocity_update_function=vfunc)
