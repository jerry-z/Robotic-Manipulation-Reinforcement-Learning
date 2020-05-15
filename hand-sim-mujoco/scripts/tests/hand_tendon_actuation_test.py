#!/usr/bin/env python3
from hand_tendon_model.hand_tendon_driven import *
import hand_sim_mj.utils.util_parser_disp as util
import numpy as np
import datetime
import ipdb
import sys

if __name__ == '__main__':
	# output terminal as test results
	now = datetime.datetime.now()
	date_time = now.strftime("%Y-%m-%d %H:%M")
	f = open("Hand tendon-driven test_"+str(date_time)+".out", 'w')
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
	hand = HandTendonDriven(config_data)

	# testing set joint interface
	fingers_q_pos = np.array([1,1,-1.2,-1.2,1.3,1.3])/180*np.pi
	hand.set_joint(fingers_q_pos)
	motor_angs_deg = np.linspace(0,10,num=10)
	print(f'Hand tendon driven test {date_time}\n')
	# test infomation hearder
	print(f'Finger joints are locked at:')
	for idx, joint_name in enumerate(hand.joint_name):
		print("	%s = %.2f [deg],"%
		(	joint_name,
			hand.finger_joint_pos[idx]*180/np.pi))
	print(f'\nFinger geometric and mechanics parameters:')
	for idx, finger in enumerate(hand.finger_models):
		print(f'	{hand.finger_model_names[idx]}:')
		print("	  r1=%.1f[mm], r2=%.1f[mm], rm=%.1f[mm], k=%.1f[N/mm]"%
			(	finger.pulley_radii_mm[0],
				finger.pulley_radii_mm[1],
				finger.pulley_radii_mm[2],
				finger.k_stiffness))
	print("")
	# loop for actuations
	for idx, motor_ang in enumerate(motor_angs_deg):
		hand.set_joint_by_name(motor_ang/180*np.pi,'virtual_main_motor_joint')
		hand.compute_joint_torque()
		print(f'Acuation {idx}:')
		print(f'	Motor information:')
		print("		Motor set angle=%.2f[deg], Motor reach angle=%.2f[deg],"
			%(	motor_ang,
				hand.main_joint_pos/np.pi*180))
		print("		Motor length=%.2f[mm], Stalled=%s"
			%(	hand.main_joint_pos*hand.main_pulley_r,
				hand.main_actuator_stalled))
		print(f'	Tension computed:')
		for idx, t in enumerate(hand.finger_tension):
			print("		[%s]	tension t=%.2f N" % 
				(hand.finger_model_names[idx],t))
		print(f'	Finger & motor joint torques:')
		for idx, tq in enumerate(hand.finger_joint_torque):
			print("		[%s]	= %.1f [Nmm]" % 
				(hand.joint_actuator_name[idx],hand.finger_joint_torque[idx]))
		print("		[%s]	= %.1f [Nmm]" % 
			(hand.main_actuator_name, hand.main_joint_torque))
	# end of test
	f.close()