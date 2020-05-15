from enum import Enum
from hand_sim_mj.robot.multi_joint_dynamics import *
from hand_tendon_model.hand_tendon_driven import *

class Environment(MultiJointDynamics):
    def __init__(self, config_data, sim_model):
        MultiJointDynamics.__init__(self, config_data, 'Environment', sim_model)


class Robot(MultiJointDynamics):
    def __init__(self, config_data, sim_model):
        # generic init
        MultiJointDynamics.__init__(self, config_data, 'Robot', sim_model)
        self.actuator_trajectory = self.ActuatorTrajectory(config_data)
        self.joint_error_eps = config_data.getfloat('Robot', 'joint_error_epsilon')
        self.actuator_trajectory_record = []

    def record_joint(self):
        MultiJointDynamics.record_joint(self)
        self.actuator_trajectory_record.append(
            deepcopy(self.actuator_trajectory))

    class ActuatorTrajectory:
        # this class computes the command actuator trajectory mostly
        def __init__(self, config_data):
            self.config = config_data
            self.actuator_name = ast.literal_eval(
                config_data['Robot']['actuator_name'])
            self.actuator_type = ast.literal_eval(
                config_data['Robot']['actuator_type'])
            self.joint_name = ast.literal_eval(
                config_data['Robot']['joint_name'])
            self.current_goal = 0
            self.status = self.Status.DISABLED
            # goals are via points on the trajectory
            self.goals = ast.literal_eval(
                config_data['Robot']['actuator_goals'])
            self.num_goals = len(self.goals)
            # interp_time => time to reach each goal
            self.interp_time = ast.literal_eval(
                config_data['Robot']['actuator_interp_time']) 
            # pause_time => pausing time after reaching each goal 
            self.pause_time = ast.literal_eval(
                config_data['Robot']['actuator_pause_time']) 
            # start_time is the starting time of each goal
            self.start_time = [0]*self.num_goals
            # before the first goal, what is the start joint positions
            self.start_joint_pos = np.nan
            # over time tolerance
            self.time_out_thresh = config_data.getfloat(
                'Robot', 'time_out_thresh')
            if config_data.has_section('Hand-tendon-driven'):
                self.init_tendon_model(config_data)
            else:
                self.tendon_model = False

        
        def init_tendon_model(self, config_data):
            self.tendon_model = HandTendonDriven(config_data)
            self.tendon_actuator_goals = ast.literal_eval(
                config_data['Hand-tendon-driven']['main_actuator_goals'])
            # obtain the joint mapping - joint index between the tendon model and mj model
            joint_idx_in_mj_list = []
            for joint_name in self.tendon_model.joint_name:
                joint_idx_in_mj_list.append(self.joint_name.index(joint_name))
            self.tendon_joint_mapping = joint_idx_in_mj_list

        class Status(Enum):
            DISABLED=0
            IN_PROCESS=1
            VIA_GOAL_REACHED=2
            STUCK=3
            FINISHED=4

        def enable(self, current_joint_state):
            if self.tendon_model:
                self.set_joint_to_tendon_model(current_joint_state)
            self.start_time[self.current_goal]=current_joint_state.time
            self.status=self.Status.IN_PROCESS
            self.start_joint_pos = current_joint_state.pos

        def update_goal(self, current_joint_state):
            if (self.current_goal<(self.num_goals-1)):
                self.current_goal = self.current_goal + 1
                self.enable(current_joint_state)
            else:
                pass

        def compute_interp_control(self, time, current_joint_state=None):
            # now, it is assumed that the actuator space and joint space are matching
            if (self.current_goal==0):
                start_position = self.start_joint_pos
            else:
                start_position = self.goals[self.current_goal-1]
            start_time = self.start_time[self.current_goal]
            goal_time = start_time + self.interp_time[self.current_goal]
            goal_position = self.goals[self.current_goal]

            # if tendon_model had been defined:
            if self.tendon_model:
                # compute the interpolated command
                if (self.current_goal==0):
                    tendon_actuator_start_pos = self.tendon_model.main_joint_pos
                else:
                    tendon_actuator_start_pos = self.tendon_actuator_goals[self.current_goal-1]
                tendon_actuator_goal = self.tendon_actuator_goals[self.current_goal]
                tendon_actuator_interp = np.interp(
                    time,[start_time,goal_time],[tendon_actuator_start_pos,tendon_actuator_goal])
                self.set_joint_to_tendon_model(current_joint_state)
                self.tendon_model.set_joint_by_name(
                    tendon_actuator_interp, self.tendon_model.main_joint_name)
                self.tendon_model.compute_joint_torque()
                torque_Nm_tendon_model_computed = self.tendon_model.finger_joint_torque/1000.0

            # interpolation
            interp_control = np.zeros(len(goal_position))
            for idx, goal_position_i in enumerate(goal_position):
                if (self.actuator_type[idx]=="position"):
                    interp_control[idx] = np.interp(
                        time,[start_time,goal_time],[start_position[idx],goal_position_i])
                elif (self.actuator_type[idx]=="motor"):
                    # in case of "motor", need to apply torque directly
                    interp_control[idx] = goal_position_i
                elif (self.actuator_type[idx]=="tendon"):
                    joint_actuator_name = self.actuator_name[idx]
                    joint_index_in_tendon_model = (
                        self.tendon_model.joint_actuator_name.index(joint_actuator_name))
                    interp_control[idx] = torque_Nm_tendon_model_computed[joint_index_in_tendon_model]
            return interp_control
        
        def get_current_goal(self):
            return self.goals[self.current_goal]

        def check_pause_time(self, time):
            # if the current time has passed the pausing time after finishing a goal
            time_plus = (
                (time - self.start_time[self.current_goal]) - 
                self.interp_time[self.current_goal])
            return (time_plus>self.pause_time[self.current_goal])
        def check_over_time(self, time):
            time_plus = (
                (time - self.start_time[self.current_goal]) - 
                self.interp_time[self.current_goal])
            return (time_plus>self.time_out_thresh)

        def set_joint_to_tendon_model(self, current_joint_state):
            finger_joint_pos = current_joint_state.pos[self.tendon_joint_mapping]
            self.tendon_model.set_joint(finger_joint_pos)

        class ActuatorTrajectoryCopy:
            def __init__(self, current_goal=0):
                self.current_goal = current_goal

        def __deepcopy__(self,memo):
            return self.ActuatorTrajectoryCopy(self.current_goal)

    def enable_actuator(self):
        self.actuator_trajectory.enable(self.joint)

    def check_if_goal_reached(self):
        joint_errors = self.actuator_trajectory.get_current_goal() - self.joint.pos
        for idx, val in enumerate(joint_errors):
                if (self.actuator_trajectory.actuator_type[idx]=="position"):
                    pass
                else:
                    # in case of "motor", need to apply torque directly
                    joint_errors[idx] = 0
        error_norm = np.linalg.norm(joint_errors)
        return (error_norm<self.joint_error_eps)

    def compute_control(self):
        if (self.check_if_goal_reached()):
            if (self.actuator_trajectory.check_pause_time(self.joint.time)):
                self.actuator_trajectory.update_goal(self.joint)
        elif (self.actuator_trajectory.check_over_time(self.joint.time)):
            self.actuator_trajectory.update_goal(self.joint)
        control_pos = self.actuator_trajectory.compute_interp_control(self.joint.time, self.joint)
        return control_pos
