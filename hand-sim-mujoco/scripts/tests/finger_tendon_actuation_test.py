#!/usr/bin/env python3
from hand_tendon_model.finger_tendon_driven import *
import hand_sim_mj.utils.util_parser_disp as util
import numpy as np
import datetime
import ipdb
import sys

if __name__ == '__main__':
	# output terminal as test results
	now = datetime.datetime.now()
	date_time = now.strftime("%Y-%m-%d %H:%M")
	f = open("Finger tendon-driven test_"+str(date_time)+".out", 'w')
	sys.stdout = f

	# read and parse the config_data
	if (len(sys.argv)==3):
		config_folder_path = sys.argv[2]
	else:
		config_folder_path=''
	if len(sys.argv)==1:
		config_file_name = 'grasping.cfg'
	else:
		config_file_name = sys.argv[1]
	config_data = util.read_config_file(config_file_name,config_folder_path)

	# test information header
	finger = FingerTendonDriven(config_data['Tendon-driven-model-finger1'])
	motor_angs_deg = np.linspace(0,10,num=10)
	q_pos = [1/180*np.pi, 1/180*np.pi, motor_angs_deg[0]/180*np.pi]
	finger.set_joint(q_pos)
	print(f'Finger tendon driven test {date_time}\n')
	print(f'Finger geometric and mechanics parameters:')
	print("	r1=%.1f[mm], r2=%.1f[mm], rm=%.1f[mm], k=%.1f[N/mm]"%(
		finger.pulley_radii_mm[0],
		finger.pulley_radii_mm[1],
		finger.pulley_radii_mm[2],
		finger.k_stiffness))
	print(f'Finger joints are locked at:')
	print("	Joint1=%.2f[deg], Joint2=%.2f[deg]"%(q_pos[0]*180/np.pi,q_pos[1]*180/np.pi))
	print(f'Finger joint names:')
	print(f'{finger.joint_actuator_name[0]}, {finger.joint_actuator_name[1]}, {finger.joint_actuator_name[2]}\n')

	# loop for actuations
	for idx, motor_ang in enumerate(motor_angs_deg):
		q_pos[2] = motor_ang/180*np.pi
		finger.set_joint(q_pos)
		t = finger.compute_tension()
		print(f'Acuation {idx}:')
		print(f'	Motor information:')
		print("		Motor angle=%.2f[deg], Motor length=%.2f[mm]"%(motor_ang,motor_ang/180*np.pi*5))
		print(f'	Tension computed:')
		print("		Tendon tension t=%.2f N"%(t))
		torque = finger.compute_joint_torque()
		print(f'	Finger & motor joint torques')
		print("		joint1=%2.f[Nmm], joint2=%2.f[Nmm], motor=%2.f[Nmm]\n"%(torque[0],torque[1],torque[2]))

	# end of test
	f.close()