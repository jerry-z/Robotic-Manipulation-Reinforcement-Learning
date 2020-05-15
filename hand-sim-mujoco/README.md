hand-sim-mujoco
=======================
Simulation code for robotic hands using mujoco-py environment

## Dependencies
* python3.6-dev, see [this page](https://stackoverflow.com/questions/43621584/why-cant-i-install-python3-6-dev-on-ubuntu16-04)
* Python3.6, see [this page](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get)
* sudo apt install libosmesa6-dev

## Set up the Environment
* Setup the virutalenv and source it  
`virtualenv -p /usr/bin/python3.6 mujoco_py36`  
`source  mujoco_py36/bin/activate`  
* Follow the install instructions on [mujoco-py wiki page](https://github.com/openai/mujoco-py)
* Setup the env var pointing to this repo (perhaps add them to `mujoco_py36/bin/activate`):  
`export HAND_SIM_MJ=/home/long/dev/hand-sim-mujoco`  
`export HAND_SIM_DATA=/home/long/dev/hand-sim-mujoco-data`  
`unset PYTHONPATH`  
`export PYTHONPATH=$PYTHONPATH:/home/long/dev/py_envs/mujoco_py36/lib/python3.6`  
`export PYTHONPATH=$PYTHONPATH:$HAND_SIM_MJ`  
