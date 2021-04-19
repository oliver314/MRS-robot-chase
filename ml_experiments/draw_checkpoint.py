import neat
import visualize

class LiveVisualReporter(neat.reporting.BaseReporter):
    def end_generation(self, config, population, species_set):
        global gen_no
        #visualize.plot_stats(stats, ylog=False, view=False, filename="gen" + str(gen_no) + "-stats-" + datetime.now().strftime("%d-%m-%Y---%H:%M:%S"))
        #visualize.plot_species(stats, view=False, filename="gen" + str(gen_no) + "-species-" + datetime.now().strftime("%d-%m-%Y---%H:%M:%S"))
        gen_no += 1


node_names = {
    -1: "P1 x",
    -2: "P1 y",
    -3: "P1 θ",
    -4: "P1 Li-FL",
    -5: "P1 Li-F",
    -6: "P1 Li-FR",
    -7: "P2 x",
    -8: "P2 y",
    -9: "P2 θ",
    -10: "P2 Li-FL",
    -11: "P2 Li-F",
    -12: "P2 Li-FR",
    -13: "P3 x",
    -14: "P3 y",
    -15: "P3 θ",
    -16: "P3 Li-FL",
    -17: "P3 Li-F",
    -18: "P3 Li-FR",
    -19: "B1 x",
    -20: "B1 y",
    -21: "B2 x",
    -22: "B2 y",
    -23: "B3 x",
    -24: "B3 y",
    0: "P1 u",
    1: "P1 ω",
    2: "P2 u",
    3: "P2 ω",
    4: "P3 u",
    5: "P3 ω",
}

cpointer = neat.Checkpointer(1)

pop = cpointer.restore_checkpoint("neat-checkpoint-336")

best_genome = pop.population[list(pop.population.keys())[0]]

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-big-robot-network')

visualize.draw_net(config, best_genome, node_names=node_names)
