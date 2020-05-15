import subprocess
import multiprocessing
import sys
import os
import ipdb
"""
Run simulations under a experiment folder

*	This script runs all simulations under
	<experiment_name> folder,
	and saves the simulation data of each experiment 
	under <experiment_name>/data folder.
*	This script also runs the evaluation of using simulation
	data of each experiment, and saves the evaluation
	result under <results>
*	This script uses sub-processes to speed up
	the simulation, and the number of sub-processes
	is determined by the number of 
	sub-folders named <batchX>
"""

NUM_RESULTS_EXPECTED=7

def get_all_config_batch_targets(experiment_folder):
	all_sub_config_dir = [] # all sub configuration folder names
	all_simu_data_targets = [] # target names under each sub-config folder
	config_dir = os.path.join(experiment_folder,'config')
	for sub_config_dir in os.listdir(config_dir):
		if os.path.isdir(os.path.join(config_dir, sub_config_dir)):
			sub_config_dir_full = os.path.join(config_dir, sub_config_dir)
			all_sub_config_dir.append(sub_config_dir_full)
			sub_dir_targets=[]
			for sub_dir_target in os.listdir(sub_config_dir_full):
				if(sub_dir_target.endswith(".cfg")):
					target_id = os.path.splitext(sub_dir_target)[0]
					sub_dir_targets.append(target_id+'.joblib')	
			all_simu_data_targets.append(sub_dir_targets)
	return all_sub_config_dir, all_simu_data_targets

def get_experiment_folders(experiment_name):
	data_root = os.environ['HAND_SIM_DATA']
	experiment_folder = os.path.join(data_root,experiment_name)
	simulation_data_folder = os.path.join(data_root,experiment_name,'data')
	results_folder = os.path.join(data_root,experiment_name,'results')
	if not os.path.isdir(simulation_data_folder):
		os.mkdir(simulation_data_folder)
	if not os.path.isdir(results_folder):
		os.mkdir(results_folder)
	return experiment_folder, simulation_data_folder, results_folder


# start subprocess to process all simulations in each sub-folder
if __name__ == '__main__':
	# analyze number of batches and target simulations per batch
	experiment_folder, simulation_data_folder, results_folder = get_experiment_folders(sys.argv[1])
	config_sub_dirs, data_targets = get_all_config_batch_targets(experiment_folder)
	# start sub-processes based on the number of batches 
	processes=[]
	for i in range (len(config_sub_dirs)):
		command_line = "python sim_robot_folder.py "+ config_sub_dirs[i]
		p = subprocess.Popen([command_line], shell=True, stdout=subprocess.DEVNULL)
		processes.append(p)

	# keep tracking the progress of each batch (sub-process)
	batch_in_process = [True for i in range(len(config_sub_dirs))]
	batch_data_targets_finished = [0 for i in range(len(config_sub_dirs))]
	# check if the simulation is finished by checking if simulation data files being generated
	while any(batch_in_process):
		total_data_targets_finished = os.listdir(simulation_data_folder)
		for i in range(len(batch_in_process)):
			if (batch_in_process[i]):
				# check how many simulation data files created as a sign of progress of all simulations
				finished_targets_batch_i_num = len(list(set(data_targets[i]) & set(total_data_targets_finished)))
				if (finished_targets_batch_i_num>batch_data_targets_finished[i]):
					print(f'  Batch {i} task update: {finished_targets_batch_i_num}/{len(data_targets[i])}')
					batch_data_targets_finished[i] = finished_targets_batch_i_num
				if (finished_targets_batch_i_num==len(data_targets[i])):
					batch_in_process[i]=False

	# all simulation data files are finished
	# below is checking if the evaluations of each simulation are done
	evaluation_in_process = True
	while evaluation_in_process:
		results_sub_folders = os.listdir(results_folder)
		if len(results_sub_folders)==len(total_data_targets_finished):
			evaluation_in_process = False
			for i in range(len(results_sub_folders)):
				results_sub_folder_content = os.listdir(os.path.join(results_folder,results_sub_folders[i]))
				if len(results_sub_folder_content)==NUM_RESULTS_EXPECTED:
					pass
				else:
					evaluation_in_process=True
	print("All simulations finished.")