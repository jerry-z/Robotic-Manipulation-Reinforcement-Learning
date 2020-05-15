#!/usr/bin/env python3
import os
import gym
import argparse
import numpy as np
import pandas as pd
import roam_gym_envs
from scripts.parser import BaselineParser
from scripts.meta_agent import MetaAgent
from success_criterion.success_criterion import SuccessCriterion
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

def test_executive():

    episodes = 50
    max_timesteps = 1000

    #Input elements for various trained policies
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--expert_policy_dir', type=str) # trained policy directory, not necessary for openloop
    arg_parser.add_argument('--type', type=str) #ars, ppo1, ppo2, openloop
    arg_parser.add_argument('--mode', type=str) #linearbias, mlp, 3finger, 2finger
    arg_parser.add_argument('--render', type=str)#on, off
    args = arg_parser.parse_args()
    mode_dict = {}
    mode_dict['mode']= args.mode

    #Parsing parameters/file to reset environment for testing objects
    categ = ['shape','x', 'y', 'rot_z', 'len', 'width', 'height','radius']
    path = os.path.join(os.environ['GRASPING_RL'],'evaluation/baseline_testset_calcs.csv' )
    csv_file = pd.read_csv(path, sep='\t', header=None, names = categ, skiprows=1)
    parser = BaselineParser(csv_file)

    #Enviornment Creation
    env_id = 'ROAMHandGraspCube-v1'
    if "ppo" in args.type:
        env = VecNormalize(DummyVecEnv([lambda: gym.make(env_id)]), norm_reward=False)
        env.set_attr('_max_episode_steps',max_timesteps)
        env.env_method('set_evaluation')
        env.load_running_average(args.expert_policy_dir)
    else:
        env = gym.make(env_id)
        env._max_episode_steps = max_timesteps
        env.env.set_evaluation()

    #Testing loop to evaluate grasps on 50 objects
    total_successes = np.zeros(episodes)
    for i in range(episodes):
        obs = env.reset()
        params = parser.get_testcase(i)

        if "ppo" in args.type:
            env.env_method('set_object',params)
            success = SuccessCriterion(env.get_attr('sim')[0])
        else:
            env.env.set_object(params)
            success = SuccessCriterion(env.env.sim)

        agent = MetaAgent(env =env, load_dir=args.expert_policy_dir, load_type=args.type, **mode_dict)  

        #Per episode simulation and evaluation
        success_array = np.zeros(max_timesteps) 
        for j in range(max_timesteps): 
            action = agent.act(obs)
            if args.type == 'openloop':
                env.env.sim.step()
            else:
                obs, reward , done, info = env.step(action)
            if args.render != 'off':
                env.render()
            success_array[j] = success.grasp_criteria()

        #Success Criterion Evaluation
        if np.sum((success_array)) >= 250:
            total_successes[i] = 1
            print("Baseline {} is a Success!".format(i), np.sum((success_array)))
        else: 
            total_successes[i] = 0
            print("Baseline {} is a Failure!".format(i), np.sum((success_array)))

    return total_successes


if __name__ == '__main__':
    test_success_array = test_executive()
    print('Individual Success Cases: ', test_success_array)
    print('Test Objects Success Rates')
    print('Cubes: ',np.sum(test_success_array[0:30]),'/30')
    print('Spheres: ',np.sum(test_success_array[30:40]), '/10')
    print('Cylinders: ',np.sum(test_success_array[40:50]),'/10')