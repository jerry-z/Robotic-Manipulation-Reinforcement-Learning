#!/usr/bin/env python3
import os
import configparser as ConfigParser
from io import StringIO
import hand_sim_mj.utils.util_parser_disp as util

# env vars defined in util under sim
SIM_ENV_VAR = util.SIM_ENV_VAR
SIM_DATA_ENV_VAR = util.SIM_DATA_ENV_VAR
PI = util.PI

# env vars defined for uncertainty study
UNCERTAINTY_ENV_VAR = 'UNCERTAINTY_HAND_SIM'

def read_config_file(config_file_name, config_folder_path=''):
	if (config_folder_path):
		path = os.path.join(config_folder_path,config_file_name)
	else:
		path = os.path.join(os.environ[UNCERTAINTY_ENV_VAR],'config/',config_file_name)
	config_data = ConfigParser.ConfigParser()
	config_data.read(path)
	return config_data

def export_config(config_data,sub_folder_name=''):
	# get experiment info and prepare folder to export
	experiment_name = config_data['Uncertainty']['experiment_name']
	current_sample_ID = config_data['Uncertainty']['current_sample_ID']
	export_folder_path = os.path.join(
		os.environ[SIM_DATA_ENV_VAR],experiment_name,
		'config',sub_folder_name)
	if (not os.path.isdir(export_folder_path)):
		os.makedirs(export_folder_path)
	# export config
	file_name = os.path.join(export_folder_path,current_sample_ID+'.cfg')
	with open(file_name, 'w') as configfile:
		config_data.write(configfile)
		configfile.close()
	print(f'Exp. config. generated at [{file_name}]')