# roam-gym-envs
Gym environments for ROAM lab hands

## Setting up the Environment
* The mujoco models are in hand-sim-mujoco, so you need to setup an environment variable to point to the path, for example:

`export HAND_SIM_MJ=$HOME/dev/hand-sim-mujoco`

## Installation
* cd to the directory

`pip install -e .`

## Usage
* For example:

`import roam_gym_envs`  
`import gym`  
`env = gym.make('ROAMHandGraspCube-v1')`  
