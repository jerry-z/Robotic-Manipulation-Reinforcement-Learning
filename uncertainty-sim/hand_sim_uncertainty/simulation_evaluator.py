from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import ast
import os
import ipdb
import joblib
import PyKDL
import numpy as np
# import from this repo
import hand_sim_mj.utils.util_parser_disp as util
from hand_sim_uncertainty.robot_simulator import *
from hand_sim_mj.robot.robot import *

def compute_pose_velocity_cartesian_eular(joint, joint_vel):
	# computes the pose and velocity (twist) based on cartesian and euler
	# the default eular sequence used is ZYX
	# compute the pose (position and orientation)
	pos = PyKDL.Vector(joint[0],joint[1],joint[2])
	rot = PyKDL.Rotation.EulerZYX(joint[3],joint[4],joint[5])
	pose = PyKDL.Frame(rot, pos)
	# compute the twist (velocity)
	lin_vel = PyKDL.Vector(joint_vel[0],joint_vel[1],joint_vel[2])
	rotation_local = PyKDL.Rotation.Identity()
	ang_vel = PyKDL.Vector()
	for i in range(3,6):
		if (i==3):
			det_rot = joint_vel[i]*rotation_local.UnitZ()
			rotation_local = rotation_local*PyKDL.Rotation.RotZ(joint[i])
		elif (i==4):
			det_rot = joint_vel[i]*rotation_local.UnitY()
			rotation_local = rotation_local*PyKDL.Rotation.RotY(joint[i])
		elif (i==5):
			det_rot = joint_vel[i]*rotation_local.UnitX()
		ang_vel = ang_vel + det_rot
	twist = PyKDL.Twist(lin_vel,ang_vel)
	return pose, twist

class SimulationEvaluator(RobotSimulatorMuJoCo):
	def __init__(self, config_file_name, config_folder_path=''):
		RobotSimulatorMuJoCo.__init__(self, config_file_name, config_folder_path)
		config_data = util.read_config_file(config_file_name,config_folder_path)
		self.config_data = config_data
		self.load_simulation_data()
		# pose represented as [x,y,z,qx,qy,qz,qw]
		self.pose_in_hand = []
		self.pose_robot = []
		self.pose_object = []
		# twist represented as [vx,vy,vz,wx,wy,wz]
		self.twist_in_hand = []
		# actuator goals are the index of the goal in the actuator goal list
		# in the format of an array of integers
		self.actuator_goals = []
		# evaluation score
		self.score=[]

	def load_simulation_data(self):
		# load the saved simulation object
		data_path = os.path.join(os.environ['HAND_SIM_DATA'],
			self.config_data['Uncertainty']['experiment_name'],
			'data',self.config_data['Uncertainty']['current_sample_ID']+'.joblib')
		try:
			sim_data = joblib.load(data_path)
		except:
			print(f'  joblib loading [{data_path}] failed')
		# load the joint_record
		self.robot.joint_record = sim_data.robot.joint_record
		self.robot.actuator_trajectory_record = sim_data.robot.actuator_trajectory_record
		self.robot.joint = sim_data.robot.joint
		self.environment.joint_record = sim_data.environment.joint_record
		self.environment.joint = sim_data.environment.joint

	def evaluate(self,export=True):
		self.compute_object_relative_pose_velocity()
		eval_parameters = ast.literal_eval(
            self.config_data['Uncertainty']['grasping_eval_param'])
		if eval_parameters['criterion-name']=='default':
			self.score = self.evaluate_default(eval_parameters['thresh'])
		elif eval_parameters['criterion-name']=='end-in-hand-position':
			self.score = self.evaluate_in_hand_position(
				eval_parameters['position_ref'],eval_parameters['r_region'])
		if export:
			self.export_results()

	def export_results(self):
		data_folder_path = os.path.join(os.environ['HAND_SIM_DATA'],
			self.config_data['Uncertainty']['experiment_name'],'results',self.config_data['Uncertainty']['current_sample_ID'])
		if (not os.path.isdir(data_folder_path)):
			os.makedirs(data_folder_path)
		data_path = os.path.join(data_folder_path,'pose_in_hand')
		np.save(data_path,self.pose_in_hand)
		data_path = os.path.join(data_folder_path,'twist_in_hand')
		np.save(data_path,self.twist_in_hand)
		data_path = os.path.join(data_folder_path,'actuator_goals')
		np.save(data_path,self.actuator_goals)
		data_path = os.path.join(data_folder_path,'pose_robot')
		np.save(data_path,self.pose_robot)
		data_path = os.path.join(data_folder_path,'pose_object')
		np.save(data_path,self.pose_object)
		data_path = os.path.join(data_folder_path,'score')
		np.save(data_path,self.score)
		data_path = os.path.join(data_folder_path,'perturb')
		np.save(data_path,np.array(
			ast.literal_eval(self.config_data['Uncertainty']['pose_purterb'])))

	def compute_object_relative_pose_velocity(self):
		# this func computes the relative pose and velocity of the object (environment) w.r.t. the hand base
		num_samples = len(self.robot.joint_record)
		self.pose_in_hand = np.zeros(shape=(num_samples,7))
		self.pose_robot = np.zeros(shape=(num_samples,7))
		self.pose_object = np.zeros(shape=(num_samples,7))
		self.twist_in_hand = np.zeros(shape=(num_samples,6))
		self.actuator_goals = np.zeros(shape=(num_samples,1))
		for i in range(num_samples):
			relative_pose, relative_twist, current_goal, pose_rob, pose_obj = (
				self.get_relative_pose_velocity(i))
			self.actuator_goals[i] = current_goal
			for j in range(3):
				self.pose_in_hand[i,j] = relative_pose.p[j]
				self.pose_robot[i,j] = pose_rob.p[j]
				self.pose_object[i,j] = pose_obj.p[j]
				self.twist_in_hand[i,j] = relative_twist.vel[j]
				self.twist_in_hand[i,j+3] = relative_twist.rot[j]
			quat = relative_pose.M.GetQuaternion()
			for j in range(3,7):
				self.pose_in_hand[i,j] = quat[j-3]
				self.pose_robot[i,j] = quat[j-3]
				self.pose_object[i,j] = quat[j-3]


	def get_relative_pose_velocity(self, record_idx):
		# get the pose and velocity of the robot based on the following hard coded joint names:
		cartesian_joint_names = [
			"palm_slide_X", 
			"palm_slide_Y", 
			"palm_slide_Z", 
			"palm_eular_Z", 
			"palm_eular_Y", 
			"palm_eular_X"]
		cartesian_joint = []
		cartesian_joint_vel = []
		for cartesian_joint_name in cartesian_joint_names:
			joint_idx = self.robot.joint_name.index(cartesian_joint_name)
			cartesian_joint.append(self.robot.joint_record[record_idx].pos[joint_idx])
			cartesian_joint_vel.append(self.robot.joint_record[record_idx].vel[joint_idx])
		rob_pose, rob_twist = compute_pose_velocity_cartesian_eular(cartesian_joint, cartesian_joint_vel)
		# get the pose and velocity of the object based on the following hard coded joint names:
		cartesian_joint_names = [
			"object_slide0", 
			"object_slide1", 
			"object_slide2", 
			"object_eular_Z",
			"object_eular_Y",
			"object_eular_X"]
		cartesian_joint = []
		cartesian_joint_vel = []
		for cartesian_joint_name in cartesian_joint_names:
			joint_idx = self.environment.joint_name.index(cartesian_joint_name)
			cartesian_joint.append(self.environment.joint_record[record_idx].pos[joint_idx])
			cartesian_joint_vel.append(self.environment.joint_record[record_idx].vel[joint_idx])
		obj_pose, obj_twist = compute_pose_velocity_cartesian_eular(cartesian_joint, cartesian_joint_vel)
		pose_relative = rob_pose.Inverse()*obj_pose
		twist_relative = obj_twist - rob_twist # the twist is writtne in world frame
		twist_relative_in_rob_base = rob_pose.M.Inverse()*twist_relative
		# get the actuator goal index for this sample time
		goal_index = self.robot.actuator_trajectory_record[record_idx].current_goal
		return pose_relative, twist_relative_in_rob_base, goal_index, rob_pose, obj_pose

	def evaluate_default(self, dist_thresh):
		# the default evaluation criterion is naiive:
		# it compares the initial and final in-hand position of the boject
		# if the position error is bigger than the dist_thresh, score=0, otherwise, score=1
		start_goal = 2
		all_index = np.where(self.actuator_goals>=start_goal)[0]
		start_index = all_index[0]
		end_index = all_index[-1:][0]
		pos_error_norm = np.linalg.norm(self.pose_in_hand[start_index][0:3] - 
			self.pose_in_hand[end_index][0:3])
		return 1.0 if pos_error_norm <dist_thresh else 0.0

	def evaluate_in_hand_position(self, position_ref, r_region):
		# evaluate the grasp based on the final object in-hand position
		# if it is outside of a spherical region specified by radius r_region, then score=0, otherwise score=1
		end_index = len(self.actuator_goals) - 1
		pos_error_norm = np.linalg.norm(self.pose_in_hand[end_index][0:3] - 
			np.array(position_ref))
		return 1.0 if pos_error_norm <r_region else 0.0
