import math
import random
from typing import List, Tuple
from genetics.Genome import Genome
from genetics.genetic_algorithm import GenomeFitness


def create_population(population_size: int, nr_of_genes: int) -> List[Genome]:
    return [
        [random.uniform(-1, 1) for _ in range(nr_of_genes)]
        for _ in range(population_size)
    ]


def linear_rank_select_parents(genomes: List[GenomeFitness], amount: int) -> List[Tuple[Genome, Genome]]:
    populationSize = len(genomes)
    cumulativeProbabilities = []
    cumulativeSum = 0

    for index in range(populationSize):
        cumulativeSum += (populationSize - index) / populationSize
        cumulativeProbabilities.append(cumulativeSum)

    selectedSpecimen = []

    while len(selectedSpecimen) < amount:
        probability = random.uniform(0, cumulativeSum)
        for index, cum_prob in enumerate(cumulativeProbabilities):
            if probability <= cum_prob:
                selectedSpecimen.append(genomes[index])
                break

    parents = []
    random.shuffle(selectedSpecimen)
    for parent1, parent2 in zip(selectedSpecimen[0::2], selectedSpecimen[1::2]):
        parents.append((parent1.genome, parent2.genome))

    return parents


def crossover(genome1: Genome, genome2: Genome, crossover_rate: float = 0.5) -> Genome:
    result = []

    for index in range(len(genome1)):
        if random.random() < crossover_rate:
            result.append(genome1[index])
        else:
            result.append(genome2[index])

    return result


def mutate(genome: Genome, mutationStrength: float, geneMutationChance: float) -> Genome:
    return [
        gene + random.gauss(0, mutationStrength) if random.random() < geneMutationChance else gene
        for gene in genome
    ]


def sigmoid(x: float):
    return (1 / (1 + math.exp(-x)) - 0.5) * 2
