import gym
import time
import numpy as np
from gym import spaces

class SeaHandGraspContextEnv(gym.Wrapper):
    # pass in information about the env's physics parameters
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.obseervation_space = spaces.Box(-np.inf, np.inf, shape=(env.observation_space.high.size + 2,))
        self.action_space = env.action_space

    def reset(self):
        # unwrap until we find base env
        baseenv = self.env
        while not hasattr(baseenv, 'objsize'):
            baseenv = baseenv.env
        self.objsize = np.array([baseenv.objsize])
        self.objangle = np.array([baseenv.objangle])
 
        obs = self.env.reset()
      
        # concatenate
        obs = np.concatenate([obs.flatten(), self.objsize.flatten(), self.objangle.flatten()])
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = np.concatenate([obs.flatten(), self.objsize.flatten(), self.objangle.flatten()])
        return obs, r, done, info
  
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

class ObservationGaussianNoiseEnv(gym.Wrapper):
    # randomly corrupt 100 * p percent of all observation with gaussian noise
    def __init__(self, env, p=0.3, sigma=0.2):
        gym.Wrapper.__init__(self, env)
        self.p = p
        self.sigma = sigma

    def reset(self):
        obs = self.env.reset()
        mask = np.ones_like(obs)
        mask[np.random.rand(*mask.shape) < self.p] = 0.0
        self.mask = mask
        return obs + (1.0 - self.mask) * np.random.randn(*mask.shape) * self.sigma

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return obs + (1.0 - self.mask) * np.random.randn(*self.mask.shape) * self.sigma, r, done, info

class ObservationMaskingEnv(gym.Wrapper):
    # randomly mask out observations in each episode
    def __init__(self, env, p=0.1):
        gym.Wrapper.__init__(self, env)
        self.p = p

    def reset(self):
        obs = self.env.reset()
        mask = np.ones_like(obs)
        mask[np.random.rand(*mask.shape) < self.p] = 0.0
        self.mask = mask
        return obs * self.mask

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return obs * self.mask, r, done, info

class MultipleEnvsEnv(gym.Wrapper):
    # randomly sample one env to execute per episode

    def __init__(self, env, envlist):
        gym.Wrapper.__init__(self, env)
        self.envlist = envlist

    def reset(self):
        # select one env
        self.envcounter = np.random.randint(0,len(self.envlist))
        self.currentenv = self.envlist[self.envcounter]
        return self.currentenv.reset()

    def step(self, action):
        return self.currentenv.step(action)

    def render(self, *args, **kwargs):
        return self.currentenv.render(*args, **kwargs)


from gym.spaces import Box
class RL2Env(gym.Wrapper):

	def __init__(self, env, num_episodes=2):
		"""each episode of this environment contains num_episodes episodes of the base environment"""
		gym.Wrapper.__init__(self, env)
		self.num_episodes = num_episodes
		self.prev_action = np.zeros(env.action_space.high.size)
		self.prev_reward = np.array([0.0])
		self.prev_done = np.array([1.0])
		self.episode_counter = 0
		self.observation_space = Box(-np.inf, np.inf, shape=(env.observation_space.high.size + self.prev_action.size + 2,))

	def reset(self):
		"""return concatenation of [obs, prev_action, prev_done]"""
		# update episode counter
		self.episode_counter = 0
		# update action and done
		self.prev_action = np.zeros(self.env.action_space.high.size)
		self.prev_reward = np.array([0.0])
		self.prev_done = np.array([1.0])
		# reset base env
		obs = self.env.reset(test=0)
		#time.sleep(3)
		return np.concatenate([obs, self.prev_action, self.prev_reward, self.prev_done])

	def step(self, action):
		"""return concatenation of [obs, prev_action, prev_done]"""		
		# step the base env
		obs, r, done, _ = self.env.step(action)
		# update action
		self.prev_action = action
		self.prev_reward = np.array([r])
		self.prev_done = np.array([1.0]) if done else np.array([0.0])

		# check done
		rl2done = False
		if done:
			# update the counter
			self.episode_counter += 1
			if self.episode_counter == self.num_episodes:
				rl2done = True # rl2 env done
			else:
				# rl2 env is not done, reset old env
				obs = self.env.reset(test=1)
				print('base env reset')
				#time.sleep(3)

		# concate and return
		return np.concatenate([obs, self.prev_action, self.prev_reward, self.prev_done]), r, rl2done, {}

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)


class StickyActionEnv(gym.Wrapper):

	def __init__(self, env, skip=5):
		"""execute an input action for multiple steps"""
		gym.Wrapper.__init__(self, env)
		self._skip = skip

	def reset(self):
		return self.env.reset()

	def step(self, action):
		total_reward = 0.0
		done = None
		for i in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			total_reward += reward
			if done:
				break
		output_ob = obs # output the last obs in the sequence. is it a good choice?

		return output_ob, total_reward, done, info

"""
e = gym.make('Pendulum-v0')
e = StickyActionEnv(e, 10)
e.reset()
done = False
while not done:
	obs,_,done,_ = e.step(e.action_space.sample())
	print(obs)
"""


