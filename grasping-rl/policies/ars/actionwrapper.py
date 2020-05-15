import numpy as np
import collections
from gym.spaces import Box

def stackstatewrapper(env, num_states):
    """
    stack multiple states together
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step = unwrapped_env.step
    unwrapped_env.orig_reset = unwrapped_env.reset

    def new_reset():
        # state recorder
        unwrapped_env.statebuffer = collections.deque(maxlen=num_states)
        for i in range(num_states):
            unwrapped_env.statebuffer.append(np.zeros(unwrapped_env.observation_space.high.size))
        obs = unwrapped_env.orig_reset()
        unwrapped_env.statebuffer.append(obs)
        return np.concatenate(unwrapped_env.statebuffer)

    def new_step(action):
        obs, rew, done, info = unwrapped_env.orig_step(action)
        unwrapped_env.statebuffer.append(obs)

        return np.concatenate(unwrapped_env.statebuffer), rew, done, info

    unwrapped_env.step = new_step
    unwrapped_env.reset = new_reset

    env.observation_space = Box(-np.inf, np.inf, shape=(unwrapped_env.observation_space.high.size * num_states,))

    return env

"""
import gym
e = gym.make('Ant-v1')
e = stackstatewrapper(e,5)
print(e.observation_space)
print(e.reset())
print(e.step(e.action_space.sample()))
"""