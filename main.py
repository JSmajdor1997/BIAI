import json
import os
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import random

from GUI.getPlotter import getPlotter, PlotConfig
from genetics.Genome import Genome
from genetics.genetic_algorithm import genetic_algorithm, EpochFinishedParams, GenomeFitness
from helpers.helpers import create_population, linear_rank_select_parents, crossover, mutate, sigmoid
from neural_network.NeuralNetworkMap import NeuralNetworkMap
import time
import gym
import matplotlib.pyplot as plt

LogsFilesCatalogPath = "./logs/"
ShowSimulation = False
# ModelToLoad = None
ModelToLoad = ""


def create_timestamped_json_file(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return open(os.path.join(directory, f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}.json"), 'w')

LogsFile = create_timestamped_json_file(LogsFilesCatalogPath)

Seed = 123
PopulationSize = 100
NrOfEpochs = 1000
EliteSize = 4
NNMap = NeuralNetworkMap(
    nr_of_inputs=27,        # observation space
    layers=[8]              # action space
)
total_nr_of_genes = NNMap.get_total_nr_of_genes()
GeneMutationChance = 1 / total_nr_of_genes
MutationStrength = 0.01

CrossoverRate = 0.4

fitnessPlotter = getPlotter("Fitness - Epochs", "epochs", "fitness score", [
    PlotConfig("Best Specimen", "r"),
    PlotConfig("Average of population", "g")
], (-500, 500))

def main():
    global ShowSimulation

    np.random.seed(Seed)
    random.seed(Seed)

    env = gym.make("Ant-v4", render_mode="human" if ShowSimulation else None, healthy_z_range=(0.4, float("inf")))

    # plt.ion()
    # fig, ax = plt.subplots()

    def onEpochStarted(epoch: int):
        print(f'EPOCH {epoch} started')
        pass

    def onEpochFinished(params: EpochFinishedParams):
        LogsFile.write('')
        json.dump(params.population[:max(EliteSize, 1)], LogsFile)

        print(f'EPOCH {params.epoch} finished | best fitness value is: {params.population[0].fitness}')

        fitnessPlotter("Best Specimen", (params.epoch, params.population[0].fitness))

        fitness = [it.fitness for it in params.population]
        fitnessPlotter("Average of population", (params.epoch, sum(fitness) / len(fitness)))
        plt.pause(0.01)

    frameCount = 0
    def calc_fitness(genome: Genome, specimenIndex: int) -> float:
        s = env.reset()
        done = False
        score = 0

        frameCount = 0
        while not done:
            frameCount += 1
            action = NNMap.getResults(genome, env.observation_space.sample(), sigmoid)
            _, reward, done, _ = tuple(env.step(action)[:4])
            score += reward

            # if(frameCount % 20 == 0):
            #     img = env.render()
            #     ax.imshow(img)

        return score


    best_fit = genetic_algorithm(
        initial_population=create_population(PopulationSize, total_nr_of_genes),
        elite_size=EliteSize,
        end_condition_predicate=lambda epoch, best_fitness, population: epoch >= NrOfEpochs,
        calc_fitness=calc_fitness,
        select_parents=linear_rank_select_parents,
        crossover=lambda a, b: crossover(a, b, CrossoverRate),
        mutate=lambda genome: mutate(genome, MutationStrength, GeneMutationChance),
        onEpochStarted=onEpochStarted,
        onEpochFinished=onEpochFinished
    )

    env.close()

    print(f"best fit = {best_fit}")

main()
LogsFile.close()