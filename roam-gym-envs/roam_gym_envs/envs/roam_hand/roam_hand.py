import numpy as np
from gym import utils, spaces
from roam_gym_envs.envs.roam_env import ROAMEnv
import ipdb
class ROAMHandEnv(ROAMEnv, utils.EzPickle):
    '''
    ROAM hand environment
    '''
    def __init__(self, model_path, nsubsteps, relative_ctrl, relative_ctrl_scale, action_normalized, obs_normalized, evaluation):
        # relative_ctrl_scale: a number in [0, 1] - relative control, this number scales the range of relative action
        # action_normalized (bool): if true, the action is in [-1, 1], if False, the action is the real action with physics meaning
        # obs_normalized (bool): if true, the observation is in [-1, 1], if False, the observation is the real observation with physics meaning

        ROAMEnv.__init__(self, model_path, nsubsteps)
        utils.EzPickle.__init__(self)
        self.evaluation = evaluation
        self._relative_ctrl = relative_ctrl
        self._relative_ctrl_scale = relative_ctrl_scale
        self._action_normalized = action_normalized
        self._obs_normalized = obs_normalized

    def set_evaluation(self):
        self.evaluation = True

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            obs (object): agent's observation of the current environment (normalized to sensor range)
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        real_obs = self._get_sensor_obs()
        additional_obs = self._get_additional_obs()
        obs = np.concatenate([real_obs, additional_obs])
        if self._obs_normalized == True:
            obs = self._real_obs_to_normalized(obs)

        if self._action_normalized == True:
            action = self._normalized_action_to_real(action)
        self._set_action(action) 
        self.sim.step()
        if self.evaluation == True:
            done = False
        else:
            done = self._check_termination()
        info = {}
        reward = self._compute_reward()
        return obs, reward, done, info

    def get_obs(self):
        real_obs = self._get_sensor_obs()
        additional_obs = self._get_additional_obs()
        obs = np.concatenate([real_obs, additional_obs])
        return obs


    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def _get_sensor_obs(self):
        raise NotImplementedError

    def _get_additional_obs(self):
        raise NotImplementedError

    def _real_obs_to_normalized(self, real_obs):
        raise NotImplementedError

    def _normalized_action_to_real(self, normalized_action):
        raise NotImplementedError

    def _set_action(self, real_action):
        raise NotImplementedError

    def _check_termination(self):
        raise NotImplementedError

    def _compute_reward(self):
        raise NotImplementedError

