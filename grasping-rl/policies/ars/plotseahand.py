import matplotlib
matplotlib.use('TkAgg')
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import numpy as np

def compute_ave(x,h):
	# compute ave with window h
	ave = []
	t = 0
	while t <= x.size - 1:
		ave.append(np.mean(x[t:t+h]))
		t += h
	return np.array(ave)

hdict = []
seeddict = [100,101,102]
colordict = ['r','b','g','m']
lrdict = [3e-5]
hdict = []
labels = []
numunits = 3
numlayers = 4
cliprange = 0.2
h = 1
entcoef = 0.01
numunits = 3
numlayers = 4
idx = 0
numstates = 1
limit = 30000000
for lr,color in zip(lrdict,colordict):
	
	# env = 'SeaHandGraspCubeFree-v0'
	env = 'LiftCubeFree-v0'

	for seed in seeddict:
		r = np.load('data/env_{}/seed_{}std_0.02stepsize_0.01policy_linearbiasnumstates_{}/reward.npy'.format(env, seed, numstates))
		t = np.load('data/env_{}/seed_{}std_0.02stepsize_0.01policy_linearbiasnumstates_{}/timesteps.npy'.format(env, seed, numstates))

		r = r[t<=limit]
		t = t[t<=limit]

		plt.plot(t,r,color='r')
		print(seed,np.mean(r[-2:]))
plt.show()
