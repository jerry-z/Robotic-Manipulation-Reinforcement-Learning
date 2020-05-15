#!/usr/bin/env
"""
Visualize results
"""
import sys
from hand_sim_uncertainty.result_visualizer import *

import ipdb

if __name__ == '__main__':
    if (len(sys.argv)==3):
        exp_name = sys.argv[2]
    else:
        exp_name=''
    plot = ResultVisualizer(sys.argv[1],exp_name)
    plot.plot_results()
    dividing_ln = '%%%%%%%%%%%%%%%%%%%%%%%%'
    print(f'{dividing_ln}\n Evaluating simulation [{exp_name}/{sys.argv[1]}] ...\n{dividing_ln}')
    print(f'{dividing_ln}\n Evaluating Simulation [{exp_name}/{sys.argv[1]}] ended.')
    print(f'{dividing_ln}')