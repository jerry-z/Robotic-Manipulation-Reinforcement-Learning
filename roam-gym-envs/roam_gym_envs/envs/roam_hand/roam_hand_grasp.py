import numpy as np
import ipdb
from gym import utils, spaces
from roam_gym_envs.envs.roam_hand.roam_hand import ROAMHandEnv


class ROAMHandGraspEnv(ROAMHandEnv):
    """Environment for ROAM hand grasping one or more object(s)"""
    def __init__(self, model_path, nsubsteps, relative_ctrl,relative_ctrl_scale, action_normalized, obs_normalized, obj_pos_lb, obj_pos_ub, obj_euler_lb, obj_euler_ub, evaluation):
        ROAMHandEnv.__init__(self, model_path, nsubsteps, relative_ctrl, relative_ctrl_scale, action_normalized, obs_normalized, evaluation)

        self._obj_pos_lb = obj_pos_lb
        self._obj_pos_ub = obj_pos_ub

        self._obj_euler_lb = obj_euler_lb
        self._obj_euler_ub = obj_euler_ub

        # define action space
        self._real_action_range = self.sim.model.actuator_ctrlrange 
        if self._action_normalized == True:
            low = -np.ones(self.sim.model.nu)
            high = np.ones(self.sim.model.nu)
        else:
            low = self._real_action_range[:, 0]
            high = self._real_action_range[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # define observation space in the derived class! (specific to the chosen objects)

    def _get_sensor_obs(self):
        hand_pos = []
        hand_trq = []
        obj_center_pos = []
        sensordata = self.sim.data.sensordata
        for name in self.sim.model.sensor_names:
            if "rob" in name and "pos" in name:
                i = self.sim.model.sensor_name2id(name)
                for j in range(self.sim.model.sensor_dim[i]):
                    hand_pos.append(sensordata[i+j])
            elif "rob" in name and ("trq" in name or "torq" in name):
                i = self.sim.model.sensor_name2id(name)
                for j in range(self.sim.model.sensor_dim[i]):
                    hand_trq.append(sensordata[i+j])
            elif "obj" in name and "pos" in name:
                i = self.sim.model.sensor_name2id(name)
                for j in range(self.sim.model.sensor_dim[i]):
                    obj_center_pos.append(sensordata[i+j])

        self._hand_pos = np.array(hand_pos)
        self._hand_trq = np.array(hand_trq)
        self._obj_center_pos = np.array(obj_center_pos)

        sensor_obs = np.concatenate([self._hand_pos, self._hand_trq, self._obj_center_pos])

        return sensor_obs

    def _get_additional_obs(self):
        # object-specific
        raise NotImplementedError

    def _real_obs_to_normalized(self, real_obs):
        # object-specific, as the dimension of obs space is depended on the additional obs
        raise NotImplementedError

    def _check_termination(self):
        obj_x = self._obj_center_pos[0]
        obj_y = self._obj_center_pos[1]

        actuator_x_range = np.array([-np.inf, np.inf])
        actuator_y_range = np.array([-np.inf, np.inf])

        for name in self.sim.model.actuator_names:
            if "rob" in name and "slide" in name and "x" in name:
                actuator_x_range = self._real_action_range[self.sim.model.actuator_name2id(name)]
            if "rob" in name and "slide" in name and "y" in name:
                actuator_y_range = self._real_action_range[self.sim.model.actuator_name2id(name)]

        if obj_x < actuator_x_range[0] or obj_x > actuator_x_range[1] or obj_y < actuator_y_range[0] or obj_y > actuator_y_range[1]:
            return True

        else:
            return False




    def _normalized_action_to_real(self, normalized_action):
        assert normalized_action.shape[0] == self._real_action_range.shape[0]

        real_action_center = (self._real_action_range[:, 1] + self._real_action_range[:, 0])/2
        real_action_half_range = (self._real_action_range[:, 1] - self._real_action_range[:, 0])/2
        real_action = real_action_center + normalized_action * real_action_half_range

        if self._relative_ctrl == True:
            real_action = real_action * self._relative_ctrl_scale

        return real_action

    def _set_action(self, real_action):
        assert real_action.shape == (self.sim.model.nu,)

        if self._relative_ctrl == True:
            self.sim.data.ctrl[:] = self._hand_pos + real_action

        else:
            self.sim.data.ctrl[:] = real_action
        
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, self._real_action_range[:, 0], self._real_action_range[:, 1])





    def reset_model(self):

        if self.evaluation == True:
            pass
        else: 
            obj_pos = np.random.uniform(self._obj_pos_lb, self._obj_pos_ub)
            obj_euler = np.random.uniform(self._obj_euler_lb, self._obj_euler_ub)

            obj_quat = euler2quat(obj_euler)
            obj_qpos = np.concatenate([obj_pos, obj_quat]) # has a dim of 7

            obj_idx = []
            for name in self.sim.model.joint_names:
                if "obj" in name:
                    obj_idx.append(self.sim.model.joint_name2id(name))

            state = self.sim.get_state()
            for i in obj_idx:
                state.qpos[self.sim.model.jnt_qposadr[i] : self.sim.model.jnt_qposadr[i]+obj_qpos.shape[0]] = obj_qpos
            self.sim.set_state(state)
            self.sim.forward()
        obs = np.concatenate([self._get_sensor_obs(), self._get_additional_obs()])
        return obs


    def set_object(self, params):
        assert self.evaluation == True
        objbody_id = []
        objgeom_id = []
        for name in self.sim.model.body_names:
            if "obj" in name:
                objbody_id.append(self.sim.model.body_name2id(name))
        for name in self.sim.model.geom_names:
            if "obj" in name:
                objgeom_id.append(self.sim.model.geom_name2id(name))

        self.sim.model.geom_size[objgeom_id,:] = params['size']
        self.sim.model.geom_type[objgeom_id]= params['type']
        self.sim.model.body_mass[objbody_id]= params['mass']
        self.sim.model.body_inertia[objbody_id,:]= params['inertia']
        
        obj_qpos = np.concatenate([params['pos'],params['quat']])
        state = self.sim.get_state()

        obj_idx = []
        for name in self.sim.model.joint_names:
            if "obj" in name:
                obj_idx.append(self.sim.model.joint_name2id(name))

        state = self.sim.get_state()
        for i in obj_idx:
            state.qpos[self.sim.model.jnt_qposadr[i] : self.sim.model.jnt_qposadr[i]+obj_qpos.shape[0]] = obj_qpos
        self.sim.set_constants()
        self.sim.set_state(state)
        self.sim.forward()



def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat