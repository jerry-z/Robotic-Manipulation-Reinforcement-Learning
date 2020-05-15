#!/usr/bin/env python3
"""
This script run a single robot simulation
"""
import sys
from hand_sim_uncertainty.robot_simulator import *
import ipdb

if __name__ == '__main__':
    if (len(sys.argv)==3):
        config_file_path = sys.argv[2]
    else:
        config_file_path=''
    if len(sys.argv)==1:
        config_file_name = 'grasping.cfg'
    else:
        config_file_name = sys.argv[1]
    dividing_ln = '%%%%%%%%%%%%%%%%%%%%%%%%'
    print(f'{dividing_ln}\nSimulation [{config_file_path}/{config_file_name}] starts ...\n{dividing_ln}')
    sim = RobotSimulatorMuJoCo(config_file_name,config_file_path)
    sim.step()
    while (not sim.check_time_limit()):
        sim.update_state()
        sim.update_robot_control()
        sim.step()
        sim.record_state()
    print(f'{dividing_ln}\n[Simulation Ended] Simulation [{config_file_path}/{config_file_name}] ended at [{sim.sim.data.time}]')
    sim.dump_data()
    print(f'{dividing_ln}')