#!/usr/bin/env python3
"""
This script evaluates a simulation and save the result of evaluation to a data folder
"""
import sys
from hand_sim_uncertainty.simulation_evaluator import *

import ipdb

if __name__ == '__main__':
    if (len(sys.argv)==3):
        config_file_path = sys.argv[2]
    else:
        config_file_path=''
    sim = SimulationEvaluator(sys.argv[1],config_file_path)
    dividing_ln = '%%%%%%%%%%%%%%%%%%%%%%%%'
    print(f'{dividing_ln}\n Evaluating simulation [{config_file_path}/{sys.argv[1]}] ...\n{dividing_ln}')
    sim.evaluate(export=False)
    print(f'{dividing_ln}\n Evaluating score = {sim.score}')
    print(f'{dividing_ln}')