import os
import ipdb
import PyKDL
import numpy as np
# for matplotlib 3D
import tkinter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import from this repo
import hand_sim_mj.utils.util_parser_disp as util

class ResultVisualizer():
	def __init__(self, data_id='1', experiment_name='test1'):
		data_folder_path = os.path.join(os.environ[util.SIM_DATA_ENV_VAR],experiment_name,'results',data_id)
		self.load_plot_data(data_folder_path)

	def load_plot_data(self,data_folder_path):
		self.pose_in_hand = np.load(os.path.join(data_folder_path,'pose_in_hand.npy')) 
		self.twist_in_hand = np.load(os.path.join(data_folder_path,'twist_in_hand.npy')) 
		self.actuator_goals = np.load(os.path.join(data_folder_path,'actuator_goals.npy')) 
		self.pose_robot = np.load(os.path.join(data_folder_path,'pose_robot.npy'))
		self.pose_object =  np.load(os.path.join(data_folder_path,'pose_object.npy'))

	def plot_results(self):
		goal_to_start_plot = 2
		# pose in hand 2D
		fig = plt.figure()
		ax1 = fig.add_subplot(221)
		goals = self.actuator_goals[:,0]
		xs = self.pose_in_hand[goals>=goal_to_start_plot,0]
		ys = self.pose_in_hand[goals>=goal_to_start_plot,1]
		zs = self.pose_in_hand[goals>=goal_to_start_plot,2]
		ax1.scatter(ys, zs, c='r', marker='o')
		ax1.set_xlabel('X Label')
		ax1.set_ylabel('Z Label')
		ax1.set_aspect('equal', 'box')
		# pose in hand
		ax2 = fig.add_subplot(222, projection='3d')
		goals = self.actuator_goals[:,0]
		xs = self.pose_in_hand[goals>=goal_to_start_plot,0]
		ys = self.pose_in_hand[goals>=goal_to_start_plot,1]
		zs = self.pose_in_hand[goals>=goal_to_start_plot,2]
		ax2.scatter(xs, ys, zs, c='r', marker='o')
		ax2.set_xlabel('X Label')
		ax2.set_ylabel('Y Label')
		ax2.set_zlabel('Z Label')
		ax2.set_aspect('equal', 'box')
		# hand pose and object pose in world
		ax2 = fig.add_subplot(223, projection='3d')
		xs = self.pose_robot[goals>=goal_to_start_plot,0]
		ys = self.pose_robot[goals>=goal_to_start_plot,1]
		zs = self.pose_robot[goals>=goal_to_start_plot,2]
		ax2.scatter(xs, ys, zs, c='b', marker='o')
		xs = self.pose_object[goals>=goal_to_start_plot,0]
		ys = self.pose_object[goals>=goal_to_start_plot,1]
		zs = self.pose_object[goals>=goal_to_start_plot,2]
		ax2.scatter(xs, ys, zs, c='m', marker='o')
		ax2.set_xlabel('X Label')
		ax2.set_ylabel('Y Label')
		ax2.set_zlabel('Z Label')
		ax2.set_aspect('equal', 'box')
		plt.show()