import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ROAMHandGraspCube-v1',
    entry_point='roam_gym_envs.envs:ROAMHandGraspCubeEnv',
    max_episode_steps=200,
)
