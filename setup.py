#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

VERSION = '0.1.0'


def load_requirements():
    """
    Loads requirements.txt
    :param:
    :return: list of requirements
    """

    with open('requirements.txt') as f:
        return [req for req in f.read().split() if req]


def run_setup():
    """
    Runs the setup to create the wheel file.
    """

    setup_params = {
        'name': 'genalg',
        'version': VERSION,
        'description': 'Genetic algorithm solver',
        'packages': ['genalg'],
        'python_requires': '>=3.7',
        'install_requires': []}
    setuptools.setup(**setup_params)


if __name__ == '__main__':
    run_setup()