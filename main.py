import json
import os
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import random

from GUI.CLI import CLI
from GUI.getPlotter import getPlotter, PlotConfig
from genetics.Genome import Genome
from genetics.genetic_algorithm import genetic_algorithm, EpochFinishedParams, GenomeFitness
from helpers.get_current_datetime import get_current_datetime
from helpers.helpers import create_population, linear_rank_select_parents, crossover, mutate, sigmoid
from logs.LogManager import LogManager
from neural_network.NeuralNetworkMap import NeuralNetworkMap
import time
import gym
import matplotlib.pyplot as plt


LogsFilesCatalogPath = "./logs/"
ShowSimulation = False
ModelToLoad = ""

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

CurrentDate = get_current_datetime()

def learn():
    global ShowSimulation

    np.random.seed(Seed)
    random.seed(Seed)

    env = gym.make("Ant-v4", render_mode="human" if ShowSimulation else None, healthy_z_range=(0.4, float("inf")))

    def onEpochStarted(epoch: int):
        print(f'EPOCH {epoch} started')
        pass

    def onEpochFinished(params: EpochFinishedParams):
        print(f'EPOCH {params.epoch} finished | best fitness value is: {params.population[0].fitness}')

        fitnessPlotter("Best Specimen", (params.epoch, params.population[0].fitness))

        logsManager.update_value(CurrentDate, {
            "genome": params.population[0].genome,
            "epoch": params.epoch,
            "fitness": params.population[0].fitness
        })

        fitness = [it.fitness for it in params.population]
        fitnessPlotter("Average of population", (params.epoch, sum(fitness) / len(fitness)))
        plt.pause(0.01)


    def calc_fitness(genome: Genome, specimenIndex: int) -> float:
        observation, info = env.reset(seed=Seed)
        done = False
        score = 0

        while not done:
            action = NNMap.getResults(genome, observation, sigmoid)
            observation, reward, done, *_ = env.step(action)
            score += reward

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


def runBestModel():
    bestModel = sorted(logsManager.read_file_as_array(), key=lambda x: x["fitness"], reverse=True)[0]

    env = gym.make("Ant-v4", render_mode="human", healthy_z_range=(0.4, float("inf")))
    observation, info = env.reset(seed=Seed)

    while True:
        action = NNMap.getResults(bestModel["genome"], observation, sigmoid)
        observation, *_ = env.step(action)


logsManager = LogManager(LogsFilesCatalogPath)

def setShowSimulation(show):
    global ShowSimulation

    ShowSimulation = show

cli = CLI({
    "Learn": [
        {
            "Show Ant": lambda: setShowSimulation(True),
            "Don show Ant": lambda: setShowSimulation(False)
        },
        {
            "With bias": lambda: learn(),
            "Without bias": lambda: learn(),
            "Rank Selection": lambda: print("Action for Answer 1.1"),
            "Stochastic Universal Sampling Selection": lambda: learn(),
            "Truncation Selection": lambda: learn(),
            "Single Point Crossover": lambda: learn(),
            "Two Point Crossover": lambda: learn(),
            "Uniform Crossover": lambda: learn(),
            "Uniform Mutation": lambda: learn(),
            "With Elite": lambda: learn()
        }
    ],
    "Run best model": runBestModel
})


cli.run()
