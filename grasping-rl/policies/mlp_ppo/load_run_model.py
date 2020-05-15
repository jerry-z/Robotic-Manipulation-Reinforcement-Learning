import gym
import roam_gym_envs
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from stable_baselines import PPO2
import sys

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



if __name__ == '__main__':

    log_str = sys.argv[1]

    env_id = 'ROAMHandGraspCube-v1'

    model = PPO2.load("logs/{}/trained_model".format(log_str))

    # render trained agent
    env = VecNormalize(DummyVecEnv([lambda: gym.make(env_id)]), norm_reward=False)
    env.load_running_average("logs/{}".format(log_str))

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()