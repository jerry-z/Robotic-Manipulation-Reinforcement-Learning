import sys
import hand_sim_mj.utils.util_parser_disp as util
import hand_sim_uncertainty.utils.util_parser_disp as util_uncertainty
import multiprocessing
import os
import ast
import numpy as np
import ipdb
"""
Generate Simulation configuration files
* 	This script generates configuration files (.cfg) 
	corresponding to variations of the base configuration 
	<grasping.cfg> to be simulated. 
*	All generated <.cfg> files are exported to a data folder
	outside of the repo and that is specified in the base 
	configuration file <grasping.cfg>
*	In addition, all generated <.cfg> files are stored in 
	different sub-folders named <batchX>, which is prepared 
	for the sub-processes to run them in parallel in the 
	next step. The number of batches should be specified 
	as equal or smaller than the number of your cpu cores 
"""

def add_perturb_to_config(sim_config_data,uncertainty_config_data,perturb=[0,0,0,0,0,0]):
	# combine the sim_config_data and uncertainty_config_data
	config_data_perturb = util.deep_copy_config_parser(sim_config_data)
	config_data_perturb['Uncertainty'] = uncertainty_config_data['Uncertainty']
	# obtain the perturbed actuator goals
	actuator_goals = ast.literal_eval(
		sim_config_data['Robot']['actuator_goals'])
	actuator_goals = apply_perturb_to_goals(actuator_goals,perturb)
	# write the perturbed goals and export
	config_data_perturb['Robot']['actuator_goals'] = str(actuator_goals)
	config_data_perturb['Uncertainty']['pose_purterb']=str(perturb)
	return config_data_perturb

def apply_perturb_to_goals(actuator_goals,perturb):	
	# this part is very specific to the control actions used in the simulations
	index_goals_to_perturb = [1,2]
	for i in index_goals_to_perturb:
		for joint_i in range(0,6):
			actuator_goals[i][joint_i] = (
				actuator_goals[i][joint_i] + perturb[joint_i])
	return actuator_goals

def generate_position_grid(sim_config_data, uncertainty_config_data):
	# based on the [pose_perturb_range] under [Uncertainty]
	# [6]-Dimensional grid is generated for experiment samples,
	# for each of the [6] dimensions, the number of grids depends on 
	# the property [pose_sample_size]
	pose_sample_size = np.array(ast.literal_eval(
		uncertainty_config_data['Uncertainty']['pose_sample_size']))
	pose_purterb_range = np.array(ast.literal_eval(
		uncertainty_config_data['Uncertainty']['pose_purterb_range']))
	BATCH_TOTAL_NUM = int(uncertainty_config_data['Uncertainty']['BATCH_TOTAL_NUM'])
	# pose_sampling_axis_indices = np.nonzero(pose_sample_size>1)[0]
	grid_coord_vectors = []
	for i in range(pose_sample_size.size):
		x_i_coord_vec = np.linspace(
			pose_purterb_range[i][0],
			pose_purterb_range[i][1],
			num=pose_sample_size[i])
		grid_coord_vectors.append(x_i_coord_vec)
	# generate 6D grid space
	xx, yy, zz, rx, ry, rz = np.meshgrid(
		grid_coord_vectors[0],
		grid_coord_vectors[1],
		grid_coord_vectors[2],
		grid_coord_vectors[3],
		grid_coord_vectors[4],
		grid_coord_vectors[5])
	pose_sampling_grid = 0
	# generate all configuration files of this 6D space to run
	max_num_per_batch = int(np.ceil(xx.size/BATCH_TOTAL_NUM))
	batch_contained = 0
	batch_id = 0
	ALL_PERTURB_norm = np.empty(shape=(0,1))
	for index in range(xx.size):
		x = xx.item(index)
		y = yy.item(index)
		z = zz.item(index)
		rot_1 = rx.item(index)
		rot_2 = ry.item(index)
		rot_3 = rz.item(index)
		perturb = [x,y,z,rot_1,rot_2,rot_3]
		print(f'  Index [{index+1}], Batch [{batch_id}]')
		print(f'    x=[{x}], y=[{y}], z=[{z}], r1=[{rot_1}], r2=[{rot_2}], r3=[{rot_3}]')
		config_data_perturb = add_perturb_to_config(sim_config_data, uncertainty_config_data, perturb)
		config_data_perturb['Uncertainty']['current_sample_id'] = str(index+1)
		util_uncertainty.export_config(config_data_perturb,'batch'+str(batch_id))
		# save all perturb
		ALL_PERTURB_norm = np.append(ALL_PERTURB_norm,np.linalg.norm(np.array(perturb)))
		# update batch number
		batch_contained = batch_contained + 1
		if (batch_contained == max_num_per_batch):
			batch_contained = 0
			batch_id = batch_id + 1
	# find the zero perturb reference
	zero_ref_index = np.argmin(ALL_PERTURB_norm)
	zero_ref_ID = str(zero_ref_index+1)
	return zero_ref_ID
if __name__ == '__main__':
    sim_config_data = util.read_config_file(sys.argv[1])
    uncertainty_config_data = util_uncertainty.read_config_file(sys.argv[2])
    zero_ref_ID = generate_position_grid(sim_config_data, uncertainty_config_data)
    print(f'The zero perturbance reference simulation config is [{zero_ref_ID}.cfg]')