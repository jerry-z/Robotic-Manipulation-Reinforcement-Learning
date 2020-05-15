import ast
import numpy as np
from copy import copy, deepcopy
import hand_sim_mj.utils.util_parser_disp as util
import ipdb

class MultiJointDynamics:
    def __init__(self, config_data, type_name, sim_model):
        # type_name is either [Robot] or [Environment]
        self.type = type_name
        joint_init_pos = ast.literal_eval(
            config_data[type_name]['joint_init_pos'])
        self.joint_name = ast.literal_eval(
            config_data[type_name]['joint_name'])
        self.joint_init = self.JointState(joint_init_pos)
        self.joint = self.JointState(joint_init_pos)
        self.joint_addr = copy(joint_init_pos)
        if config_data.has_option(type_name,'joint_init_vel'):
            joint_init_vel = ast.literal_eval(
                config_data[type_name]['joint_init_vel'])    
            self.joint_addr_vel = copy(joint_init_vel)
            self.initialize_from_config(joint_init_pos, sim_model, joint_init_vel)
        else:
            self.joint_addr_vel = copy(joint_init_pos)
            self.initialize_from_config(joint_init_pos, sim_model)
        self.joint_record = []

    def initialize_from_config(self, joint_init_pos, sim_model, joint_init_vel=None):
        self.joint_init.set_pos(joint_init_pos)
        if joint_init_vel:
            self.joint_init.set_vel(joint_init_vel)
        self.update_joint_addr(sim_model)

    class JointState:
        def __init__(self, place_holder):
            self.pos = self.format_zeros(place_holder)
            self.vel = self.format_zeros(place_holder)
            self.time = 0.0

        def set_time(self, time):
            self.time = time

        def set_pos_idx(self, idx, value):
            if (isinstance(value,list)):
                self.pos[idx] = np.array(value)
            else:
                # this includes cases {'float', 'np.array'}
                self.pos[idx] = value

        def set_vel_idx(self, idx, value):
            self.vel[idx] = value

        def set_pos(self, joint_pos):
            for idx, val in enumerate(joint_pos):
                self.set_pos_idx(idx, val)

        def set_vel(self, joint_vel):
            for idx, val in enumerate(joint_vel):
                self.set_vel_idx(idx, val)

        def format_zeros(self, place_holder):
            joint_zeros = np.zeros(len(place_holder))
            for idx, val in enumerate(place_holder):
                if (isinstance(val,list)):
                    joint_zeros = joint_zeros.astype(object)
                    joint_zeros[idx] = np.zeros(len(val))
            return joint_zeros

    def update_joint_addr(self,sim_model):
        print(f'{self.type} Joint address update from model:')
        for idx, joint_name in enumerate(self.joint_name):
            joint_addr = sim_model.get_joint_qpos_addr(joint_name)
            joint_addr_vel = sim_model.get_joint_qvel_addr(joint_name)
            if (isinstance(joint_addr,tuple)):
                for sub_idx, joint_addr_i in enumerate(range(joint_addr[0],joint_addr[1],1)):    
                    self.joint_addr[idx][sub_idx] = joint_addr_i
                    util.print_joint_actuator_info(
                        joint_name, joint_addr_i, "")
                for sub_idx, joint_addr_i in enumerate(range(joint_addr_vel[0],joint_addr_vel[1],1)):    
                    self.joint_addr_vel[idx][sub_idx] = joint_addr_i
                    util.print_joint_actuator_info(
                        joint_name+"_vel", joint_addr_i, "")
            else:
                self.joint_addr[idx] = joint_addr
                self.joint_addr_vel[idx] = joint_addr_vel
                util.print_joint_actuator_info(
                    joint_name, joint_addr, "")


    def read_init_state_from_config(self, sim_state):
        print(self.type + " Joint init pos read from config:")
        for idx, joint_name in enumerate(self.joint_name):
            joint_addr = self.joint_addr[idx]
            if (isinstance(joint_addr,list)):
                for sub_idx, joint_addr_i in enumerate(joint_addr):
                    sim_state.qpos[joint_addr_i] = self.joint_init.pos[idx][sub_idx]
                    util.print_joint_actuator_info(
                        self.joint_name[idx], joint_addr_i, sim_state.qpos[joint_addr_i])
            else:
                sim_state.qpos[joint_addr] = self.joint_init.pos[idx]
                util.print_joint_actuator_info(
                    joint_name, joint_addr, sim_state.qpos[joint_addr])
        return sim_state


    def get_joint(self, sim_state):
        self.joint.set_time(sim_state.time)
        for idx, joint_name in enumerate(self.joint_name):
            joint_addr = self.joint_addr[idx]
            if (isinstance(joint_addr,list)):
                # meaning this joint has more than 1 DoF
                # fetching all joint position values
                joint_pos_value = np.zeros(len(joint_addr))
                for sub_idx, joint_addr_i in enumerate(joint_addr):
                    joint_pos_value[sub_idx] = sim_state.qpos[joint_addr_i]
                # fetching all joint velocities
                joint_addr_vel = self.joint_addr_vel[idx]
                joint_vel_value = np.zeros(len(joint_addr_vel))
                for sub_idx, joint_addr_i in enumerate(joint_addr_vel):
                    joint_vel_value[sub_idx] = sim_state.qvel[joint_addr_i]
            else:
                joint_pos_value = sim_state.qpos[joint_addr]
                joint_vel_value = sim_state.qvel[joint_addr]
            self.joint.set_pos_idx(idx,joint_pos_value)
            self.joint.set_vel_idx(idx,joint_vel_value)

    def record_joint(self):
        self.joint_record.append(deepcopy(self.joint))