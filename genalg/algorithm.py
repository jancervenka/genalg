# !/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import logging
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

P_MUTATION = 0.01
N_GENERATION = 1000
TOURNAMENT_SIZE = 50
POPULATION_SIZE = 200
PROBLEM_SIZE = 20
GRID_LIMIT = 100
LOGGING_FORMAT = '%(asctime)s : %(levelname)s : %(message)s'


def init_logging():
    """
    Sets up logging.
    """

    logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)


class Problem:
    """
    Class defining the TSP.
    """

    def __init__(self, cities):
        """
        Initiates the problem.

        :param cities: list of city locations as 2D coordinates
                       `[(20, 1), (40, 32), (1, 51)]`, must be unique
        """

        if len(set(cities)) == len(cities):
            self.cities = cities
            self.size = len(cities)
        else:
            raise ValueError('Cities must be unique')

    @staticmethod
    def _get_random_location(grid_limit):
        """
        Creates a random city location.

        :param grid_limit: maximum coordinate value
        :return: tuple of coordinates e.g. `(20, 1)`
        """

        return (random.randint(0, grid_limit - 1), random.randint(0, grid_limit - 1))

    @classmethod
    def generate_random_problem(cls, size=PROBLEM_SIZE, grid_limit=GRID_LIMIT):
        """
        Creates a problem by randomly generating list of cities.

        :param size: number of cities
        :param grid_limit: maximum coordinate value
        :return: `Problem` instance
        """

        cities = []

        while len(cities) < size:
            city = cls._get_random_location(grid_limit)

            while city in cities:
                city = cls._get_random_location(grid_limit)

            cities.append(city)

        return Problem(cities)


class Solution:
    """
    Class defining a solution to a TSP problem.
    """

    def __init__(self, problem, path=None):
        """
        Initiates the solution.

        :param problem: `Problem` instance
        :param path: list of integers defining in which order to visit the cities,
                     each integer is an index of a city in the `problem.cities` list,
                     e.g. `[1, 0, 3, 2]`. If `None`, random solution is generated.
        """

        self._problem = problem

        if path:
            self.path = path
        else:
            self.path = list(range(len(self._problem.cities)))
            random.shuffle(self.path)

        dists = [euclidean(
            self._problem.cities[self.path[i]],
            self._problem.cities[self.path[i + 1]]) for i in range(len(self.path) - 1)]

        self.fitness = sum(dists)

    def mutate(self):
        """
        Mutates the solution by randomly swapping order of cities in the `self.path`,
        e.g. from `[1, 0, 3, 2]` to `[0, 2, 3, 1]`.

        :return: new `Solution` instance
        """

        path = self.path
        for i in range(len(path)):
            if random.random() < P_MUTATION:
                swap = random.randint(0, len(path) - 1)
                temp = path[swap]
                path[swap] = path[i]
                path[i] = temp

        return Solution(self._problem, path=path)


class Population:
    """
    Class defining the population of solution. One population contains
    one generation.
    """

    def __init__(self, problem, population_size=POPULATION_SIZE):
        """
        Initiates the population by producing a list of solutions to a problem.

        :param problem: `Problem` instance
        :param population_size: number of solutions in the population
        """

        self._problem = problem
        self._solutions = [Solution(problem) for _ in range(population_size)]
        self._population_size = population_size

    def _select_one_parent(self, tournament_size=TOURNAMENT_SIZE):
        """
        Creates a tournament by selecting number of solutions from the population.
        Parent is then selected as the best solution in the tournament.

        :param tournament_size: number of solutions entering the tournament
        :return: best solution from the tournament, `Solution` instance
        """

        tournament = random.sample(self._solutions, tournament_size)
        return min(tournament, key=lambda x: x.fitness)

    def _crossover(self):
        """
        Produces a new solution from two parents by applying crossover.
        Crossover takes parts of the solution from both parents to create
        a new solution with better fitness.

        :return: `Solution` instance
        """

        parents = []

        # FIXME: make sure that both parents are not the same solution
        parents = [self._select_one_parent() for _ in range(2)]

        start = random.randint(0, self._problem.size - 2)
        end = random.randint(start, self._problem.size)

        co_path = [-1] * self._problem.size
        co_path[start:end] = parents[0].path[start:end]
        unused_cities = [city for city in parents[1].path if city not in co_path]

        co_path[:start] = unused_cities[:start]
        # unused_cities is smaller than co_path, unused_cities[start:] picks the rest
        co_path[end:] = unused_cities[start:]

        return Solution(self._problem, path=co_path)

    def crossover_all(self):
        """
        Creates a new generation using crossover.
        """

        self._solutions = [self._crossover() for _ in range(self._population_size)]

    def mutate_all(self):
        """
        Mutates every solution in the new generation.
        """

        self._solutions = [solution.mutate() for solution in self._solutions]

    def get_fittest(self):
        """
        Finds the best solution in the population

        :return: `Solution` instance
        """

        return min(self._solutions, key=lambda x: x.fitness)


class GeneticAlgorithm:
    """
    Class implementing genetic algorithm.
    """

    def __init__(self, problem=None, population_size=POPULATION_SIZE, random_state=0):
        """
        Initiates the class.

        :param problem: `Problem` instance, random problem is created if `None`
        :param population_size: number of solutions in the population
        :param random state: initial RNG state
        """

        random.seed(random_state)
        self._problem = problem if problem else Problem.generate_random_problem()
        self._population = Population(self._problem, population_size=POPULATION_SIZE)
        self._current_fitness = None
        self.evolution = [self._population._solutions[0]]

    def _evolve(self):
        """
        Produces new generation.
        """

        self._population.crossover_all()
        self._population.mutate_all()
        self.evolution.append(self._population.get_fittest())
        self._current_fitness = self.evolution[-1].fitness

    def compute(self, n_generation=N_GENERATION, es_delta=1e-3, es_patience=5):
        """
        Runs the full evolution.

        :param n_generation: number of generations (evolution steps)
        :param es_delta: fitness function delta threshold for early stopping
        :param es_patience: early stopping patience (in number of generations)
        """

        patience_count = 0
        for i in range(n_generation):

            prev_fitness = self._current_fitness
            self._evolve()
            logging.info('Generation {:04d}. Fitness: {:.4f}.'.format(i, self._current_fitness))

            if i and prev_fitness - self._current_fitness < es_delta:
                patience_count += 1
            else:
                patience_count = 0

            if patience_count == es_patience:
                logging.info('Early stopping.')
                break

        improvement = self.evolution[0].fitness / self.evolution[-1].fitness
        logging.info('The found solution is {:.2f}% better than the inital guess.'.format(
            improvement * 100))

        return self._population.get_fittest()


def visualize_solution(evolution, file_path=None):
    """
    Creates a plot comparing solution from the first and last generation.

    :param evolution: list of `Solution` objectts
    :param file_path: where the figure will be stored
    """

    cities = evolution[0]._problem.cities

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, solution in enumerate([evolution[0], evolution[-1]]):
        x = [cities[k][0] for k in solution.path]
        y = [cities[k][1] for k in solution.path]

        ax[i].plot(x, y, 'r.-')
        ax[i].set_title('Generation {}, fitness: {:.2f}'.format(
            len(evolution) if i else 0, solution.fitness))

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()


def visualize_fitness(evolution, file_path=None):
    """
    Creates a plot showing fitness of each generation.

    :param evolution: list of `Solution` objectts
    :param file_path: where the figure will be stored
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    x = list(range(len(evolution)))
    y = list(map(lambda x: x.fitness, evolution))
    ax.plot(x, y, 'r')
    ax.set_xlabel('generation [-]')
    ax.set_ylabel('fitness [-]')
    ax.set_title('Evolution Progress')

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()


if __name__ == '__main__':
    init_logging()
    try:
        ga = GeneticAlgorithm()
        ga.compute(es_patience=50)
        # visualize_solution(ga.evolution, file_path='img/problem_20.png')
        # visualize_fitness(ga.evolution, file_path='img/evolution_20.png')
    finally:
        logging.shutdown()
