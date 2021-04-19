from __future__ import print_function
import neat
from simulator import *
import multiprocessing
from datetime import datetime
import pickle
import visualize


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    vfunc = create_velocity_function(net)

    genome.fitness = run_simulation(headless=True, velocity_update_function=vfunc)

    print(genome.fitness)

    return genome.fitness


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


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-2:
        return np.zeros_like(v)
    return v / n


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


gen_no = 0


class LiveVisualReporter(neat.reporting.BaseReporter):
    def end_generation(self, config, population, species_set):
        global gen_no
        visualize.plot_stats(stats, ylog=False, view=False, filename="gen" + str(gen_no) + "-stats-" + datetime.now().strftime("%d-%m-%Y---%H:%M:%S"))
        visualize.plot_species(stats, view=False, filename="gen" + str(gen_no) + "-species-" + datetime.now().strftime("%d-%m-%Y---%H:%M:%S"))
        gen_no += 1


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-big-robot-network')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

checkpointer = neat.Checkpointer(1)

p.add_reporter(checkpointer)

stats = neat.StatisticsReporter()
p.add_reporter(stats)

p.add_reporter(LiveVisualReporter())

# Run until a solution is found.
pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
winner = p.run(pe.evaluate)

with open("winner-" + datetime.now().strftime("%d-%m-%Y---%H:%M:%S"), "wb") as f:
    pickle.dump(winner, f, 2)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

vfunc = create_velocity_function(winner_net)

run_simulation(headless=False, velocity_update_function=vfunc)
