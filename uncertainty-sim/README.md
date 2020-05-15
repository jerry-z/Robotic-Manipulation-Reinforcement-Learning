# uncertainty-sim
Work on uncertainty quantification for simulation

## Set up the Environment
* Setup the virutalenv and source it  
`virtualenv -p /usr/bin/python3.6 mujoco_py36`  
`source  mujoco_py36/bin/activate`  
* Follow the install instructions on [mujoco-py wiki page](https://github.com/openai/mujoco-py)
* Setup the env var pointing to this repo (perhaps add them to `mujoco_py36/bin/activate`):  
`export HAND_SIM_MJ=/home/long/dev/hand-sim-mujoco`  
`export HAND_SIM_DATA=/home/long/dev/hand-sim-mujoco-data`  
`export UNCERTAINTY_HAND_SIM=/home/long/dev/uncertainty-sim/hand_sim_uncertainty`
`unset PYTHONPATH`  
`export PYTHONPATH=$PYTHONPATH:/home/long/dev/py_envs/mujoco_py36/lib/python3.6`  
`export PYTHONPATH=$PYTHONPATH:$HAND_SIM_MJ`  
`export PYTHONPATH=$PYTHONPATH:$UNCERTAINTY_HAND_SIM`  

## [Optional] PyKDL in Python3.6 Virtualenv
[PyKDL](https://github.com/orocos/orocos_kinematics_dynamics) is a powerfult tool for real-time computation of kinematics and dynamics. We use it with MuJoCo simulator. To do this, we need to set it up in Python3.6 Virtualenv.
### Create virtualenv ###
* ***Comment*** the ros source line in your .bashrc, and then open a new terminal, i.e.:  
`# source /opt/ros/kinetic/setup.bash`
* create virtualenv & source  
`virtualenv -p /usr/bin/python3.6 mujoco_py36`  
`source mujoco_py36/bin/activate`
### build SIP from source ###
* make sure your virtualenv is loaded during this buiding process
* download & unzip  from [this page](https://www.riverbankcomputing.com/software/sip/download/)  
* build according to [this page](https://www.riverbankcomputing.com/static/Docs/sip/installation.html)   
`cd /path/to/sip/folder`  
`python configure.py`  
`make -j8`  
`make install` # use sudo make install if it fails  
* run the following command to see if the sip is installed in your virtualenv  
`which sip`
### build PyKDL from source ###
* make sure your virtualenv is loaded during this building process  
`git clone git@github.com:roamlab/orocos_kinematics_dynamics.git`  
`cd <path/to/orocos_kinematics_dynamics>`  
* build & install orocos_kdl
`cd orocos_kdl`  
`mkdir build && cd build`  
`ccmake -DCMAKE_BUILD_TYPE=Release ..`  
`make -j4`  
`sudo make install`  
* build PyKDL  
`cd path/to/orocos_kinematics_dynamics/python_orocos_kdl`  
`mkdir build && cd build`  
`ccmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3 ..`  
`make -j4`  
* copy PyKDL to virtualenv  
`cp PyKDL.so /home/long/dev/py_envs/venv/lib/python3.6m/site-packages`
