source activate mujoco
export env=ROAMHandGraspCube-v1
#for seed in 100 101 102
#for seed in 101 102 103
for seed in 100 101 102
do
#for policy in linearbias
for policy in mlp
#for policy in toeplitz
do
#for numstates in 10 20
for numstates in 1 
do
    python ars_stack.py --policy_type $policy --numstates $numstates --n_workers 4 -nd 32 -n 5000 --env_name $env --seed $seed -s 0.01 -std 0.02 &
done
done
done
source deactivate 
