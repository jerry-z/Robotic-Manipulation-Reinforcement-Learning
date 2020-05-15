import numpy as np
from scripts.test_policies import OpenLoopPolicy, MLPPolicy, LinearBiasPolicy
import hand_sim_mj.utils.util_parser_disp as util
from stable_baselines import PPO1, PPO2

#Wraps + loads the various policies  
class MetaAgent(object):

	def __init__(self, env=None, load_dir=None, load_type=None, **kwargs):
		self.algorithm = load_type

		if self.algorithm == 'ars':
			params = np.load(load_dir+'params1.npy')
			policy_params={'ob_dim':23, 'ac_dim':12, 'ob_filter':'NoFilter','hsize':2, 'numlayers':32}
	
			if kwargs["mode"] == 'mlp':
				self.agent = MLPPolicy(policy_params)
				self.agent.load(params)
			elif kwargs["mode"] == 'linearbias':
				self.agent = LinearBiasPolicy(policy_params)
				self.agent.load(params)
			else:
				raise NotImplementedError

		elif self.algorithm == 'openloop': 
			if kwargs["mode"] == '2finger':
				config_data = util.read_config_file('gym_roam_hand_2fin_grasping_baseline.cfg','')
			elif kwargs["mode"] =='3finger':
				config_data = util.read_config_file('roam_grasping_3fin.cfg','')
			else:
				raise NotImplementedError
			self.agent = OpenLoopPolicy(config_data,env)

		elif self.algorithm == 'ppo1':
			self.agent = PPO1.load("ppo1_roam")

		elif self.algorithm == 'ppo2':
			self.agent = PPO2.load("{}/trained_model".format(load_dir))

		else:
			raise NotImplementedError

	def act(self, obs):
		if self.algorithm == 'ppo1' or self.algorithm == 'ppo2':
			action, _states = self.agent.predict(obs, deterministic=True)
			return action
		else:
			return self.agent.test_act(obs)

