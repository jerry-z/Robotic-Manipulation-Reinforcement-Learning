#!/usr/bin/env python3
"""
This script runs simulations that are specified by all configuration files under a folder.
"""
import sys
from hand_sim_uncertainty.robot_simulator import *
from hand_sim_uncertainty.simulation_evaluator import *
import os

def sim_robot_config_file(config_file_name,config_folder_path):
    dividing_ln = '%%%%%%%%%%%%%%%%%%%%%%%%'
    print(f'{dividing_ln}\nSimulation [{config_folder_path}/{config_file_name}] starts ...\n{dividing_ln}')
    sim = RobotSimulatorMuJoCo(config_file_name, config_folder_path)
    while (not sim.check_time_limit()):
        sim.step()
        sim.update_state()
        sim.update_robot_control()
        sim.record_state()
    print(f'{dividing_ln}\n[Simulation Ended] Simulation [{config_folder_path}/{config_file_name}] ended at [{sim.sim.data.time}]')
    sim.dump_data()
    print(f'{dividing_ln}')

def eval_robot_config_file(config_file_name,config_folder_path):
    dividing_ln = '%%%%%%%%%%%%%%%%%%%%%%%%'
    print(f'{dividing_ln}\n Evaluating simulation [{config_folder_path}/{config_file_name}] ...\n{dividing_ln}')
    sim = SimulationEvaluator(config_file_name,config_folder_path)
    sim.evaluate()
    print(f'{dividing_ln}\n Evaluating Simulation [{config_folder_path}/{config_file_name}] ended.')
    print(f'{dividing_ln}')

if __name__ == '__main__':
    config_folder_path = sys.argv[1]
    file_names = []
    for file_name in os.listdir(config_folder_path):
        if(file_name.endswith(".cfg")):
            sim_robot_config_file(file_name, config_folder_path)
            eval_robot_config_file(file_name, config_folder_path)
            file_names.append(file_name)