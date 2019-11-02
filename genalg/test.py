# !/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from unittest import TestCase, main
from .algorithm import Problem, Solution, Population


class TestProblem(TestCase):
    """
    Tests `genalg.algorithm.Problem`.
    """

    def test_cities_unique(self):
        """
        Tests that cities are always unique.
        """

        cities = [(30, 1), (30, 30)]
        _ = Problem(cities)

        cities = [(30, 20), (30, 20)]
        with self.assertRaises(ValueError):
            _ = Problem(cities)


class TestSolution(TestCase):
    """
    Tests `genalg.algorithm.Solution`.
    """

    def setUp(self):
        """
        Test set up.
        """

        random.seed(0)
        cities = [(20, 20), (20, 21), (20, 22)]
        problem = Problem(cities)
        self._solution = Solution(problem, path=[0, 1, 2])

    def test_fitness(self):
        """
        Tests solution fitness calculation.
        """

        self.assertEqual(self._solution.fitness, 2)

    def test_mutation(self):
        """
        Tests `genalg.algorithm.Solution.mutate`.
        """

        new_solution = self._solution.mutate()
        self.assertSetEqual(set(new_solution.path), {0, 1, 2})


class TestPopulation(TestCase):
    """
    Tests `genalg.algorithm.Population`.
    """

    def test_crossover(self):
        """
        Tests `genalg.algorithm.Population._crossover`.
        """

        random.seed(0)
        problem = Problem.get_random_problem(size=20)
        population = Population(problem=problem, population_size=100)

        crossover_solution = population._crossover()
        self.assertEqual(len(set(crossover_solution.path)), len(crossover_solution.path))
        self.assertSetEqual(set(crossover_solution.path), set(range(20)))


if __name__ == '__main__':
    main()
