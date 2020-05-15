import numpy as np
import ast
import ipdb

class FingerTendonDriven:
	def __init__(self, config_section=None):
		self.config=config_section
		self.load_finger_param()
		self.init_finger_model()
		
	def load_finger_param(self):
		self.q_pos_natrual_L = np.array(ast.literal_eval(self.config['q_pos_natrual_L']))
		self.k_stiffness = self.config.getfloat('tendon_stiffness_N_mm')
		self.pulley_radii_mm = np.array(ast.literal_eval(self.config['pulley_radii_mm']))
		self.joint_actuator_name = ast.literal_eval(self.config['joint_actuator_name'])
		self.joint_name = ast.literal_eval(self.config['joint_name'])
		# q_dir_sign is the direction sign of joint angles, 
		# because some joints increases the tendon wrapping lengths when rotating in "+" dir while others don't  
		q_dir_sign = self.config.get('q_dir_sign')
		if q_dir_sign:
			self.q_dir_sign = np.array(ast.literal_eval(q_dir_sign))
		else:
			self.q_dir_sign = np.array([1,1,1])
		# similarly, tau_dir_sign is the direction sign of joint torques
		tau_dir_sign = self.config.get('tau_dir_sign')
		if tau_dir_sign:
			self.tau_dir_sign = np.array(ast.literal_eval(tau_dir_sign))
		else:
			self.tau_dir_sign = np.array([1,1,1])
		
	def init_finger_model(self):
		# compute tendon extension offset
		# the definition of extension constant is found in Long Wang's memo on tendon modeling
		signed_pulley_radii_mm = np.multiply(self.pulley_radii_mm, self.q_dir_sign)
		self.ext_constant = - np.dot(signed_pulley_radii_mm, self.q_pos_natrual_L)
		self.q_pos = np.array(self.q_pos_natrual_L)

	def set_joint(self, q_pos):
		self.q_pos = np.array(q_pos)

	def set_joint_by_name(self, q_pos, joint_name):
		index = self.joint_name.index(joint_name)
		self.q_pos[index] = q_pos

	def compute_tension(self):
		signed_pulley_radii_mm = np.multiply(self.pulley_radii_mm, self.q_dir_sign)
		extension = np.dot(signed_pulley_radii_mm, self.q_pos) + self.ext_constant
		if extension>0:
			t = self.k_stiffness*extension
		else:
			t=0.0
		return t

	def compute_joint_torque(self):
		tension = self.compute_tension()
		signed_pulley_radii_mm = np.multiply(self.pulley_radii_mm, self.tau_dir_sign)
		torque = tension*signed_pulley_radii_mm
		return torque