'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def reset(self):
        pass


class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        #self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)
        self.weights = np.random.randn(self.ac_dim, self.ob_dim)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
        

class LinearBiasPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob> + b. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim + 1), dtype = np.float64)

    def act(self, ob, update=True):
        if update:
            ob = self.observation_filter(ob, update=self.update_filter)
        else:
            ob = self.observation_filter(ob, update=False)
        return np.dot(self.weights[:,:self.ob_dim], ob) + self.weights[:,self.ob_dim]

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        print('mu',mu,'std',std)
        #aux = np.asarray([self.weights, mu, std])
        #return aux
        return self.weights, mu, std


from policyutils import tanh, relu, fclayer, ortho_init, lstmlayer
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

    def act(self, ob, update=True):
        if update:
            ob = self.observation_filter(ob, update=self.update_filter)
        else:
            ob = self.observation_filter(ob, update=False)
        x = ob
        for layer in self.layers:
            #print(x.shape, layer.w.shape, layer.b.shape)
            x = layer(x)
        return x

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

    def get_weights_plus_stats(self):     
        mu, std = self.observation_filter.get_stats()
        print('mu',mu,'std',std)
        #aux = np.asarray([self.weights, mu, std])
        #return aux
        return self.weights, mu, std


class LSTMPolicy(Policy):

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.make_lstm_weights(policy_params['ob_dim'], policy_params['ac_dim'], policy_params['nh'])

    def make_lstm_weights(self, ob_dim, ac_dim, nh):
        self.lstmlayer = lstmlayer(ob_dim, nh, init_scale=0.01)
        self.forwardlayer = fclayer(nin=nh, nout=ac_dim, act=lambda x:x, init_scale=0.01)
        self.offsets = [0, self.lstmlayer.get_weights().size, self.forwardlayer.get_weights().size]
        self.offsets = np.cumsum(self.offsets)
        self.nh = nh

    def reset(self):
        c, h = np.zeros(self.nh), np.zeros(self.nh)
        self.state = np.concatenate([c, h])

    def act(self, ob, update=True):
        if update:
            ob = self.observation_filter(ob, update=self.update_filter)
        else:
            ob = self.observation_filter(ob, update=False)
        h, self.state = self.lstmlayer(self.state, ob)
        # use h to compute policy
        return self.forwardlayer(h)

    def get_weights(self):
        return np.concatenate([self.lstmlayer.get_weights(), self.forwardlayer.get_weights()]).copy()
    
    def update_weights(self, weights):
        self.lstmlayer.update_weights(weights[self.offsets[0]:self.offsets[1]])
        self.forwardlayer.update_weights(weights[self.offsets[1]:self.offsets[2]])
        self.weights = weights.copy()

    def get_weights_plus_stats(self):     
        mu, std = self.observation_filter.get_stats()
        print('mu',mu,'std',std)
        return self.weights, mu, std   

"""
params = {'nh':8, 'ob_dim':10, 'ac_dim':5, 'ob_filter':'MeanStdFilter'}
policy = LSTMPolicy(params)

#ww = policy.get_weights()
#print(policy.offsets)
#policy.update_weights(ww)
#print(policy.get_weights_plus_stats())

policy.reset()
for _ in range(100):
    x = np.random.randn(10)
    a = policy.act(x)
    print(a)
"""




"""
params = {'hsize':10, 'ob_dim':10, 'ac_dim':5, 'numlayers':2, 'ob_filter':'MeanStdFilter'}
policy = MLPPolicy(params)

ww = policy.get_weights()
print(policy.offsets)
policy.update_weights(ww)
print(policy.get_weights_plus_stats())

x = np.random.randn(10)
policy.act(x)
"""



