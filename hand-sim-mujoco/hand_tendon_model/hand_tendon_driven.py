import numpy as np
import ast
import ipdb
from hand_tendon_model.finger_tendon_driven import *
from copy import copy

class HandTendonDriven:
	def __init__(self, config_data):
		self.main_joint_pos = np.nan
		self.main_joint_torque = np.nan
		self.main_actuator_stalled = False
		self.main_actuator_stalled_pos = np.nan
		self.finger_joint_pos = np.nan
		self.finger_joint_torque = np.nan
		self.finger_tension = np.nan
		self.joint_actuator_name = []
		self.joint_name = []
		self.joint_finger_index = []
		self.finger_models = []
		self.load_hand_parameters(config_data)
		self.init_finger_models(config_data)
		
	def load_hand_parameters(self, config_data):
		self.main_actuator_name = config_data['Hand-tendon-driven']['main_actuator_name']
		self.main_joint_name = config_data['Hand-tendon-driven']['main_joint_name']
		self.main_actuator_max_torque = float(config_data['Hand-tendon-driven']['main_actuator_max_torque'])
		self.main_pulley_r = float(config_data['Hand-tendon-driven']['main_pulley_r_mm'])

	def init_finger_models(self, config_data):
		self.finger_model_names = ast.literal_eval(
			config_data['Hand-tendon-driven']['finger_models'])
		for idx, finger_model_name in enumerate(self.finger_model_names):
			self.finger_models.append(FingerTendonDriven(config_data[finger_model_name]))
			for actuator_name in self.finger_models[idx].joint_actuator_name:
				if (actuator_name!=self.main_actuator_name):
					self.joint_actuator_name.append(actuator_name)
					self.joint_finger_index.append(idx) # to which model this joint belongs to
			for joint_name in self.finger_models[idx].joint_name:
				if (joint_name!=self.main_joint_name):
					self.joint_name.append(joint_name)
		self.finger_joint_pos = np.zeros(shape=(len(self.joint_name)))
		self.finger_joint_pos[:] = np.nan
		self.finger_joint_torque = np.zeros(shape=(len(self.joint_name)))
		self.finger_joint_torque[:] = np.nan
		self.finger_tension = np.zeros(shape=(len(self.finger_model_names)))
		self.finger_tension[:] = np.nan
		self.set_joint_by_name(
			float(config_data['Hand-tendon-driven']['q_init']), self.main_joint_name)

	def set_joint_by_name(self, q_value, joint_name):
		if joint_name==self.main_joint_name:
			self.main_joint_pos = q_value
			for finger in self.finger_models:
				finger.set_joint_by_name(q_value, joint_name)
		else:
			index_hand_joint_list = self.joint_name.index(joint_name)
			finger_model_index = self.joint_finger_index[index_hand_joint_list]
			self.finger_models[finger_model_index].set_joint_by_name(q_value,joint_name)
			self.finger_joint_pos[index_hand_joint_list] = q_value


	def set_joint(self, q_pos, joint_name=None):
		# this func accepts q_pos to be a vector
		# with a corresponding list of joint names (a default is given)
		if not joint_name:
			joint_name = copy(self.joint_name)
		for idx, q_value in enumerate(q_pos):
			self.set_joint_by_name(q_value, joint_name[idx])	

	def compute_joint_torque(self):
		self.main_joint_torque=0.0
		for finger_number, finger in enumerate(self.finger_models):
			joint_torques = finger.compute_joint_torque()
			self.finger_tension[finger_number] = finger.compute_tension()
			# assign all finger torques according to actuator names
			for idx, torque in enumerate(joint_torques):
				if finger.joint_actuator_name[idx]==self.main_actuator_name:
					self.main_joint_torque= torque + self.main_joint_torque
				else:
					finger_joint_index = self.joint_actuator_name.index(finger.joint_actuator_name[idx])
					self.finger_joint_torque[finger_joint_index] = torque
		# checking for motor stall
		if self.main_actuator_stalled:
			# if already stalled last time
			if self.main_joint_torque<self.main_actuator_max_torque:
				self.main_actuator_stalled=False
			else:
				self.set_joint_by_name(self.main_actuator_stalled_pos,self.main_joint_name)
				self.main_actuator_stalled=False
				self.compute_joint_torque()
		else:
			if self.main_joint_torque>self.main_actuator_max_torque:
				self.redistribute_joint_torque()
				self.main_actuator_stalled=True
				self.main_actuator_stalled_pos = self.main_joint_pos

	def redistribute_joint_torque(self):
		# when main joint actuator torque exceeds
		# re-calculate the new tensions, and scale the torques using new/old tensions
		new_tension = (
			(self.finger_tension/np.sum(self.finger_tension))*
			(self.main_actuator_max_torque/self.main_pulley_r))
		torque_scale_factor = np.divide(new_tension, self.finger_tension) 
		for idx, joint_finger_index in enumerate(self.joint_finger_index):
			self.finger_joint_torque[idx] = (
				self.finger_joint_torque[idx]*torque_scale_factor[joint_finger_index])