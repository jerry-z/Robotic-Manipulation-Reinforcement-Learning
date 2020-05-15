"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
    python run_policy.py ../trained_policies/Humanoid-v1/policy_reward_11600/lin_policy_plus.npz Humanoid-v1 --render \
            --num_rollouts 20
"""
import numpy as np
import gym
import actionwrapper
#import roboschool

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str)
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert rollouts')
    args = parser.parse_args()

    print('loading and building expert policy')
    lin_policy = np.load(args.expert_policy_file)
    mode = 'linearbias'
    if mode == 'linearbias':
        obsize = lin_policy[0].shape[1]-1
        w = lin_policy[0][:,:obsize]
        b = lin_policy[0][:,-1]    
    elif mode == 'mlp':
        from policy import MLPPolicy
        policy = MLPPolicy({'hsize':2, 'numlayers':32, 'ob_dim':23, 'ac_dim':12})
        policy.update_weights(lin_policy[0].flatten())
    else:
        raise NotImplementedError



    # mean and std of state vectors estimated online by ARS. 
    mean = lin_policy[1]
    std = lin_policy[2]
        
    env = gym.make(args.envname)
    env = actionwrapper.stackstatewrapper(env, 1)

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        env.render()
        done = False
        totalr = 0.
        steps = 0
        for t in range(10000):
            observations.append(obs)
            #if t%50 == 0:
            normobs = (obs - mean) / std
            action = policy.act(normobs)
            #action = np.dot(w, (obs - mean)/std) + b
            actions.append(action)        
            obs, r, done, _ = env.step(action)
            done = False
            env.render()
            #print(r)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            #if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
            #if steps >= env.spec.timestep_limit:
            #    break
        returns.append(totalr)
        print(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('quantile return', np.percentile(returns,25),np.percentile(returns,75))
    print('std of return', np.std(returns))
    
if __name__ == '__main__':
    main()
