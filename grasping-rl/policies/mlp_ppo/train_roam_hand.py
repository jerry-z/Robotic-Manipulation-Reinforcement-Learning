import gym
import roam_gym_envs
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from datetime import datetime
import tensorflow as tf
from stable_baselines import bench, logger

import signal

keyboard_interrupt = False

def keyboardInterruptHandler(signal, frame):
    '''
    redefine the keyboardInterrupt handler to let the training finish one loop and then exit and then continue
    '''
    global keyboard_interrupt
    keyboard_interrupt = True

original_sigint_handler = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, keyboardInterruptHandler)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env
    set_global_seeds(seed)
    return _init


def stop_by_keyboard(_locals, _globals):
    if keyboard_interrupt == True:
        return False
    else:
        return True


if __name__ == '__main__':
    env_id = 'ROAMHandGraspCube-v1'
    num_cpu = 8
    vec_norm_env = VecNormalize(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method="spawn"), norm_reward=True)

    policy_kwargs = dict(act_fun=tf.tanh, net_arch=[dict(vf=[64,] * 4, pi=[64,] * 4)])

    now = datetime.now()
    log_str = now.strftime("%m-%d-%Y-%H-%M-%S")

    model = PPO2(policy="MlpPolicy", policy_kwargs=policy_kwargs, env=vec_norm_env, learning_rate=lambda f: 1e-4 * f, ent_coef=1e-2, n_steps = 256, nminibatches=8, noptepochs=8, cliprange=0.2, cliprange_vf=0.2, tensorboard_log = 'logs', verbose=1)
    
    model.learn(total_timesteps=int(1e7), seed=0, log_interval=100, tb_log_name=log_str, callback = stop_by_keyboard)
    signal.signal(signal.SIGINT, original_sigint_handler)

    model.save("logs/{}_1/trained_model".format(log_str))
    vec_norm_env.save_running_average("logs/{}_1".format(log_str))


    # render trained agent
    env = VecNormalize(DummyVecEnv([lambda: gym.make(env_id)]), norm_reward=True)
    env.load_running_average("logs/{}_1".format(log_str))

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()