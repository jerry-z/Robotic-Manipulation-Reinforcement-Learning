from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os
import ipdb
import joblib

# import from this repo
import hand_sim_mj.utils.util_parser_disp as util
from hand_sim_mj.robot.robot import *
class RobotSimulatorMuJoCo:
    def __init__(self, config_file_name, config_folder_path=''):
        # reading config file and mujoco general info
        config_data = util.read_config_file(config_file_name,config_folder_path)
        model_xml = config_data['MuJoCo']['model_xml']
        model_xml_path = os.path.join(os.environ[util.SIM_ENV_VAR],'xml/',model_xml)
        model = load_model_from_path(model_xml_path)
        self.sim = MjSim(model)
        self.render = config_data['MuJoCo']['render']
        # create robot and environment
        self.robot = Robot(config_data,self.sim.model)
        self.environment = Environment(config_data,self.sim.model)
        # simulator parameters
        self.param = {
            "mj_xml": model_xml_path,
            "total_time_limit": float('Inf')
            }
        # initialize and update from the configuration file
        self.initialize_from_config(config_data)
        # initialize render
        self.init_render()

    def initialize_from_config(self, config_data):
        # extract the init states from the configuration file and assign them to init_state
        init_state = self.sim.get_state()
        if not config_data.has_option('Robot','ignore_init'):
            init_state = self.robot.read_init_state_from_config(init_state)
        if not config_data.has_option('Environment','ignore_init'):
            init_state = self.environment.read_init_state_from_config(init_state)
        # set the init_state to the simulator
        self.sim.set_state(init_state)
        self.sim.forward()
        print("Simulator init states are set.")
        self.update_state()
        self.robot.enable_actuator()
        # update total time to stop the simulation if specified
        total_time_limit = config_data.getfloat('MuJoCo', 'total_time_limit')
        if (total_time_limit>0):
            self.param.update({'total_time_limit': total_time_limit})
        # update the experiment name and the sample ID if defined
        self.param.update(record_sampling_decimation=1)         
        self.param.update(record_sampling_idx=1)
        if (config_data.has_section('Uncertainty')):
            current_sample_id = config_data['Uncertainty']['current_sample_id']
            if (int(current_sample_id)>0):
                self.param.update({ 
                    'experiment_name' : config_data['Uncertainty']['experiment_name'],
                    'current_sample_id': current_sample_id
                    })
            if ('record_sampling_decimation' in config_data['Uncertainty']):
                self.param.update(
                    record_sampling_decimation=int(config_data['Uncertainty']['record_sampling_decimation']))

    def init_render(self):
        # [optional] init render
        if (self.render=="yes"):
            self.viewer = MjViewer(self.sim)
            print("viewer initialized")

    def step(self):
        self.sim.step()
        if (self.render=="yes"):
            self.viewer.render()

    def update_state(self):
        current_state = self.sim.get_state()
        self.robot.get_joint(current_state)
        self.environment.get_joint(current_state)

    def update_robot_control(self):
        ctrl_pos = self.robot.compute_control()
        for i, joint_name in enumerate(ctrl_pos):
            self.sim.data.ctrl[i]=ctrl_pos[i]

    def record_state(self):
        if (self.param['record_sampling_idx']==self.param['record_sampling_decimation']):
            self.robot.record_joint()
            self.environment.record_joint()
            self.param.update(record_sampling_idx=1)
        else:
            self.param.update(record_sampling_idx=self.param['record_sampling_idx']+1)

    def check_time_limit(self):
        if (self.sim.data.time<self.param['total_time_limit']):
            return False
        else:
            return True

    def dump_data(self):
        # this func dumps data only, because the model can be loaded using the xml of the simulation.
        if ('experiment_name' in self.param.keys()):
            # prepare save path
            export_folder_path = os.path.join(os.environ[util.SIM_DATA_ENV_VAR],
                self.param['experiment_name'],'data')
            if (not os.path.isdir(export_folder_path)):
                os.makedirs(export_folder_path)
            # save the simulator object without the MuJoCo model
            self.sim=[]
            file_name  = os.path.join(export_folder_path,self.param['current_sample_id']+'.joblib')
            joblib.dump(self, file_name)
            print(f'Simulation result saved at [{file_name}]')
