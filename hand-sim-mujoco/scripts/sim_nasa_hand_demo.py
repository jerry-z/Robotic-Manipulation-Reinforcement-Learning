#!/usr/bin/env python3
"""
Simulation of Nasa Hand
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import hand_sim_mj.utils.util_parser_disp as util
import os
import time
# constants
# load the dynamic model
model_xml_path = os.path.join(os.environ[util.SIM_ENV_VAR],'xml/nasa_hand/main.xml')
model = load_model_from_path(model_xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
modder = TextureModder(sim)
# set initial joint valuse
# sim.data.set_joint_qpos('palm_slide_X', 0.001)
# sim.data.set_joint_qpos('palm_slide_Y', 0.001)
# sim.data.set_joint_qpos('palm_slide_Z', 0.001)
# A = PyKDL.Rotation.RotX(0/180*util.PI)
# q = A.GetQuaternion()
# sim.data.set_joint_qpos('palm_ball_joint', q)
sim.data.set_joint_qpos('finger1_roll_joint', 1.0)
sim.data.set_joint_qpos('finger1_prox_joint', 0.0)
sim.data.set_joint_qpos('finger1_dist_joint', 0.0)
sim.data.set_joint_qpos('finger2_roll_joint', -1.0)
sim.data.set_joint_qpos('finger2_prox_joint', 0.0)
sim.data.set_joint_qpos('finger2_dist_joint', 0.0)
sim.data.set_joint_qpos('thumb_prox_joint', 0.0)
sim.data.set_joint_qpos('thumb_dist_joint', 0.0)

viewer.render()
time.sleep(2.5)

# set the joint control goals
sim.data.ctrl[0]= 0.05
sim.data.ctrl[1]= 0.05
sim.data.ctrl[2]= 0.05
sim.data.ctrl[3]= 3.14/4
sim.data.ctrl[4]= 3.14/4
sim.data.ctrl[5]= 0.00
sim.data.ctrl[6]= 0.00
sim.data.ctrl[7]= 1.50
sim.data.ctrl[8]= 1.50
sim.data.ctrl[9]= 0.00
sim.data.ctrl[10]=-1.50
sim.data.ctrl[11]=-1.50
sim.data.ctrl[12]= 1.50
sim.data.ctrl[13]=1.50

while True:
    viewer.render()
    sim.step()