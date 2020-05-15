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
#seeddict = [100,101,102]
seeddict = [100,101,102]#,101]
#seeddict = [102]
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
#env = 'StickyLiftCubeFree-v0'
#env='ROAMHandGraspCube-v1'
env='ROAMHandGraspCube-v1'
for lr,color in zip(lrdict,colordict):
	
	#env = 'roboschoolhalfcheetah-v1'

	for seed in seeddict:
		r = np.load('data/env_{}/seed_{}std_0.02stepsize_0.01policy_linearbiasnumstates_1/reward.npy'.format(env, seed))
		t = np.load('data/env_{}/seed_{}std_0.02stepsize_0.01policy_linearbiasnumstates_1/timesteps.npy'.format(env, seed))

		plt.plot(t,r,color='r')
		print(seed,np.max(r))
plt.xlabel('time stetps')
plt.ylabel('cumulative reward')
plt.title('env_{}'.format(env))
plt.show()
