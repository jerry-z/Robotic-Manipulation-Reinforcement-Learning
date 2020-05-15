# Evolutionary Algorithm (Blackbox Optimization) for ROAM Hand

The code base is built on top of ARS [Augmented Random Search](https://github.com/modestyachts/ARS). 

## Dependencies

Need to install Ray for distributed computing
```bash
pip install ray 
```

## Examples
To run the code with linear policy
```bash
python ars.py --policy_type linearbias -nd 100 --env_name Swimmer-v1 --seed 100
```
