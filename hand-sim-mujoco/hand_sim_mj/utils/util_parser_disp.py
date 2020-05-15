#!/usr/bin/env python3
import os
import configparser as ConfigParser
from io import StringIO
SIM_ENV_VAR = 'HAND_SIM_MJ'
SIM_DATA_ENV_VAR="HAND_SIM_DATA"
PI = 3.141592653589793238462643383279502884197169399375105821
def read_config_file(config_file_name, config_folder_path=''):
	if (config_folder_path):
		path = os.path.join(config_folder_path,config_file_name)
	else:
		path = os.path.join(os.environ[SIM_ENV_VAR],'config/',config_file_name)
	config_data = ConfigParser.ConfigParser()
	config_data.read(path)
	return config_data

def print_joint_actuator_info(joint_name, joint_addr, value):
    if (value==''):
        print(f'  Jnt. [{joint_name}] = [{str(joint_addr)}] ')
    else:
        print(f'  Jnt. [{joint_name}] [{str(joint_addr)}] = [{str(value)}]')

def deep_copy_config_parser(base_config):
	# Create a deep copy of the configuration object
	config_string = StringIO()
	base_config.write(config_string)
	# We must reset the buffer to make it ready for reading.        
	config_string.seek(0)        
	new_config = ConfigParser.ConfigParser()
	new_config.readfp(config_string)
	return new_config