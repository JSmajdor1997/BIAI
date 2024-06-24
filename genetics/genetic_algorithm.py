from typing import List, Callable, Tuple,Optional, Any, Dict
import multiprocessing
from genetics.Genome import Genome

class GenomeFitness:
    genome: Genome
    fitness: float

    def encode(self):
        return self.__dict__

    def __init__(self, genome: Genome, fitness: float):
        self.genome = genome
        self.fitness = fitness
        pass

class EpochFinishedParams:
    epoch: int
    population: List[GenomeFitness]

    def __init__(self, epoch: int, population: List[GenomeFitness]):
        self.epoch = epoch
        self.population = population
        pass

def genetic_algorithm(
        initial_population: List[Genome],
        elite_size: int,
        end_condition_predicate: Callable[[int, GenomeFitness, List[Genome]], bool],
        calc_fitness: Callable[[Genome, int], float],
        select_parents: Callable[[List[GenomeFitness], int], List[Tuple[Genome, Genome]]],
        crossover: Callable[[Genome, Genome], Genome],
        mutate: Callable[[Genome], Genome],
        onEpochStarted: Optional[Callable[[int], None]],
        onEpochFinished: Optional[Callable[[EpochFinishedParams], None]],
) -> Optional[GenomeFitness]:
    population: List[Genome] = initial_population
    epoch = 0
    best_fitness: Optional[GenomeFitness] = None

    while not end_condition_predicate(epoch, best_fitness, population):
        epoch += 1

        if onEpochStarted is not None:
            onEpochStarted(epoch)

        # Simulationto o
        genome_fitness: List[GenomeFitness] = []

        def printVal(value):
            return ("hello " * (value))

        pool_obj = multiprocessing.Pool()
        ans = pool_obj.map(printVal, range(0, 4))
        print(ans)
        pool_obj.close()
        for index, genome in enumerate(population):
            genome_fitness.append(GenomeFitness(genome, calc_fitness(genome, index)))

        descending_by_fitness = sorted(genome_fitness, key=lambda x: x.fitness, reverse=True)

        best_fitness = descending_by_fitness[0]

        # Selection (linear rank selection with elitism)
        elite = descending_by_fitness[:elite_size]

        new_parents = select_parents(descending_by_fitness, len(initial_population) - elite_size)

        new_children = [e.genome for e in elite] + []
        while len(new_children) < len(population):
            for p in new_parents:
                new_children.append(mutate(crossover(p[0], p[1])))

        if onEpochFinished is not None:
            onEpochFinished(EpochFinishedParams(epoch, descending_by_fitness))

    return best_fitness
