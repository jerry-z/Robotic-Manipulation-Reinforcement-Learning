# Evaluation
Evaluation scripts for the trained or pre-defined policies.

# Example Shell Commands:

OpenLoop, 3 Finger: python testing_executive.py --type=openloop --type=3finger

ARS: python testing_executive.py --type=ars --mode=linearbias --expert_policy_dir=/home/Jerry/trained_models/logs/ars/ --render=off

PPO2: python testing_executive.py --type=ppo2 --expert_policy_dir=/home/Jerry/ROAM_RL/trained_models/logs/ppo2/ --render=off
