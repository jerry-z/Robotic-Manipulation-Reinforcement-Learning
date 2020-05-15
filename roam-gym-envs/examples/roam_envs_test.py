import gym
import roam_gym_envs
import numpy as np

env = gym.make("ROAMHandGraspCube-v1")
observation = env.reset()
for _ in range(10000):
  env.render()
  normalized_action = env.action_space.sample()
  observation, reward, done, info = env.step(normalized_action)

  if done:
    observation = env.reset()
env.close()