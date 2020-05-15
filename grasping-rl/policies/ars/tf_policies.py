from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

import numpy as np
from filter import get_filter

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        
        #with tf.variable_scope("obfilter"):
        #    self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        #obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz = ob  # normalized on the outside
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
        
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))            
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        #stochastic = tf.placeholder(dtype=tf.bool, shape=())
        #ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        #ac = self.pd.sample()
        ac = self.pd.mode()
        #self._act = U.function([stochastic, ob], [ac, self.vpred])
        self._act = U.function([ob], [ac, self.vpred])

    #def act(self, stochastic, ob):
    #    ac1, vpred1 =  self._act(stochastic, ob[None])
    #    return ac1[0], vpred1[0]
    def act(self, ob):
        ac1, vpred1 = self._act(ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []


class ARS_MLPPolicy(object):
    """
    TF MLP Policy for evolutionary computation
    """
    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.policy = MlpPolicy('pi', policy_params['ob_space'], policy_params['ac_space'], hid_size=64, num_hid_layers=2)
        all_var_list = self.policy.get_trainable_variables()
        var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        self.get_flat = U.GetFlat(var_list)
        self.set_from_flat = U.SetFromFlat(var_list)
        sess = tf.get_default_session()
        U.initialize()
        #print('sess',sess)
        #sess.run(tf.global_variables_initializer())
        self.weights = self.get_flat()
        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        self.set_from_flat(self.weights)
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        ac, vpred = self.policy.act(ob)
        return ac

    def copy(self):
        raise NotImplementedError

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    