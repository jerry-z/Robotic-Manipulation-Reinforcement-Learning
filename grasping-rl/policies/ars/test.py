import gym

e=gym.make('ROAMHandGraspCube-v1')

for _ in range(100):
	d = False
	ob = e.reset()
	rsum = 0
	while not d:
		a=e.action_space.sample()
		_,r,d,_ = e.step(a)
		rsum += r
	print(rsum)
