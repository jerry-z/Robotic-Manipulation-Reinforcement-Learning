'''
Policy class for computing action from weights and observation vector. 
'''

import numpy as np
from scripts.policyutils import tanh, relu, fclayer, ortho_init, lstmlayer
from hand_sim_mj.robot.robot import *

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)
        self.biases = np.empty(0)
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def load(self, params):
        pass

    def get_weights(self):
        return self.weights

    def test_act(self, ob):
        raise NotImplementedError
  
    def copy(self):
        raise NotImplementedError

    def reset(self):
        pass

class OpenLoopPolicy(object):
    def __init__(self, config_data, env):
        self.env = env
        self.robot = Robot(config_data,env.env.sim.model)

        self.init_state = self.env.env.sim.get_state()
        init_state = self.robot.read_init_state_from_config(self.init_state)
        self.env.env.sim.set_state(init_state)
        self.env.env.sim.forward()

        current_state = self.env.env.sim.get_state()
        self.robot.get_joint(current_state)

        self.robot.enable_actuator()

    def test_act(self, obs):
        #self.env.env.sim.forward()
        #self.env.env.sim.step()
        current_state = self.env.env.sim.get_state()
        self.robot.get_joint(current_state)
        ctrl_pos = self.robot.compute_control()
        #return ctrl_pos
        for i, joint_name in enumerate(ctrl_pos):
            self.env.env.sim.data.ctrl[i]=ctrl_pos[i]
        #print('obs',obs.shape,'act',self.env.env.sim.data.ctrl.shape)

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """
    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        #self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)
        self.weights = np.random.randn(self.ac_dim, self.ob_dim)
        self.mean = None
        self.std = None

    def test_act(self,obs):
        normobs = (obs - self.mean) / self.std 
        return np.dot(self.weights, normobs) 

class LinearBiasPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob> + b. 
    """
    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim + 1), dtype = np.float64)
        self.mean = None
        self.std = None
        self.biases = None

    def act(self, ob, update=True):
        normob = (ob - self.mean) / (self.std + 1e-8)
        return np.dot(self.weights[:,:self.ob_dim], normob) + self.weights[:,self.ob_dim]

    def load(self, params):
        obsize = params[0].shape[1]-1
        w = params[0][:,:obsize]
        self.biases = params[0][:,-1]
        self.mean = params[1]
        self.std = params[2]    
        self.update_weights_test(w)
        
    def update_weights_test(self, new_weights):
        self.weights = new_weights
        return

    def test_act(self,obs):
        normobs = (obs - self.mean) / self.std 
        return np.dot(self.weights, normobs) + self.biases


class MLPPolicy(Policy):

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        hsize = policy_params['hsize']
        numlayers = policy_params['numlayers']
        self.make_mlp_weights(policy_params['ob_dim'], policy_params['ac_dim'], policy_params['hsize'], policy_params['numlayers'])

    def make_mlp_weights(self, ob_dim, ac_dim, hsize, numlayers):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hsize = hsize
        self.numlayers = numlayers
        self.layers = []
        self.offsets = [0]
        for i in range(numlayers):
            if i == 0:
                layer = fclayer(nin=ob_dim, nout=hsize, act=tanh)
                self.offsets.append(ob_dim * hsize + hsize)
            else:
                layer = fclayer(nin=hsize, nout=hsize, act=tanh)
                self.offsets.append(hsize * hsize + hsize)
            self.layers.append(layer)
        finallayer = fclayer(nin=hsize, nout=ac_dim, act=lambda x:x, init_scale=0.01)
        self.layers.append(finallayer)
        self.offsets.append(hsize * ac_dim + ac_dim)
        self.offsets = np.cumsum(self.offsets)

    def test_act(self,obs):
        #ipdb.set_trace()
        normobs = (obs - self.mean) / self.std 
        return self.act(normobs)#np.dot(self.weights, normobs) + self.biases

    def get_weights(self):
        w = []
        for layer in self.layers:
            w.append(layer.get_weights())
        return np.concatenate(w)

    def update_weights(self, weights):
        idx = 0
        for i,j in zip(self.offsets[:-1],self.offsets[1:]):
            params = weights[i:j]
            self.layers[idx].update_weights(params)
            idx += 1
        self.weights = weights.copy()

    def load(self, params):
        self.mean = params[1]
        self.std = params[2] 
        self.update_weights(params[0].flatten())
        


# class LSTMPolicy(Policy):

#     def __init__(self, policy_params):
#         Policy.__init__(self, policy_params)
#         self.make_lstm_weights(policy_params['ob_dim'], policy_params['ac_dim'], policy_params['nh'])

#     def make_lstm_weights(self, ob_dim, ac_dim, nh):
#         self.lstmlayer = lstmlayer(ob_dim, nh, init_scale=0.01)
#         self.forwardlayer = fclayer(nin=nh, nout=ac_dim, act=lambda x:x, init_scale=0.01)
#         self.offsets = [0, self.lstmlayer.get_weights().size, self.forwardlayer.get_weights().size]
#         self.offsets = np.cumsum(self.offsets)
#         self.nh = nh

#     def reset(self):
#         c, h = np.zeros(self.nh), np.zeros(self.nh)
#         self.state = np.concatenate([c, h])

#     def act(self, ob, update=True):
#         if update:
#             ob = self.observation_filter(ob, update=self.update_filter)
#         else:
#             ob = self.observation_filter(ob, update=False)
#         h, self.state = self.lstmlayer(self.state, ob)
#         # use h to compute policy
#         return self.forwardlayer(h)

#     def get_weights(self):
#         return np.concatenate([self.lstmlayer.get_weights(), self.forwardlayer.get_weights()]).copy()
    
#     def update_weights(self, weights):
#         self.lstmlayer.update_weights(weights[self.offsets[0]:self.offsets[1]])
#         self.forwardlayer.update_weights(weights[self.offsets[1]:self.offsets[2]])
#         self.weights = weights.copy()

#     def get_weights_plus_stats(self):     
#         mu, std = self.observation_filter.get_stats()
#         print('mu',mu,'std',std)
#         return self.weights, mu, std   
