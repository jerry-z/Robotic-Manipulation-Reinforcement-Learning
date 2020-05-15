import numpy as np
from gym import spaces
from roam_gym_envs.envs.roam_hand.roam_hand_grasp import ROAMHandGraspEnv
import ipdb

class ROAMHandGraspCubeEnv(ROAMHandGraspEnv):
    '''Environment that specifies which object the ROAM hand grasps'''
    def __init__(self):
        ROAMHandGraspEnv.__init__(self, 'xml/roam_hand/roam_hand_grasp_block.xml', nsubsteps=10, relative_ctrl=False, relative_ctrl_scale=0.6, 
            action_normalized=True, obs_normalized=True,
            obj_pos_lb = np.array([-0.3, -0.3, 0.05]), obj_pos_ub = np.array([0.3, 0.3, 0.1]), obj_euler_lb = -np.pi/2*np.ones(3), obj_euler_ub = np.pi/2*np.ones(3),evaluation=False)

        sensor_obs_range = []
        for i in range(self.sim.model.nsensor):
            for j in range(self.sim.model.sensor_dim[i]):
                sensor_obs_range.append([-self.sim.model.sensor_cutoff[i], self.sim.model.sensor_cutoff[i]])
        sensor_obs_range = np.array(sensor_obs_range)
        additional_obs_range = self._get_additional_obs_range()

        self._real_obs_range = np.concatenate((sensor_obs_range, additional_obs_range), axis=0)

        # The action space initialization is in the parent class because the dimension is always the number of actuators
        # But observation space needs to be initialized here, because its dimension is up to user's choice, and it is a bad practice to use polymorphism in init function

        if self._obs_normalized == True:
            obs_dim = self._real_obs_range.shape[0]
            low = -np.ones(obs_dim)
            high = np.ones(obs_dim)
        else:
            low = self._real_obs_range[:, 0]
            high = self._real_obs_range[:, 1]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)


    def _get_additional_obs(self):
        # currently empty, fill this function if necessary
        additional_obs = np.empty(0)
        return additional_obs

    def _get_additional_obs_range(self):
        # currently empty, fill this function if necessary
        # called in init function, so make sure this function does not use any member variable initialized after that call
        obs_dim = self._get_additional_obs().shape[0]
        additional_obs_range = np.empty((obs_dim, 2))
        return additional_obs_range

    def _real_obs_to_normalized(self, real_obs):
        assert real_obs.shape[0] == self._real_obs_range.shape[0]

        real_obs_center = (self._real_obs_range[:, 1] + self._real_obs_range[:, 0])/2
        real_obs_half_range = (self._real_obs_range[:, 1] - self._real_obs_range[:, 0])/2

        normalized_obs = (real_obs - real_obs_center) / real_obs_half_range
        
        return normalized_obs


    def _compute_reward(self):
        # currently only works for single hand and single obj
        # currently only dense reward

        for name in self.sim.model.body_names:
            if "palm" in name:
                hand_center_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(name), :]
                # hand_center_pos[2] -= 0.072
            elif "obj" in name:
                obj_trans_vel = self.sim.data.body_xvelp[self.sim.model.body_name2id(name), :]
                obj_rot_vel = self.sim.data.body_xvelr[self.sim.model.body_name2id(name), :]
                obj_vel = np.concatenate([obj_trans_vel, obj_rot_vel])
        hand_obj_vec = self._obj_center_pos - hand_center_pos
        hand_obj_dist = np.linalg.norm(hand_obj_vec)
        obj_floor_dist = self._obj_center_pos[2]

        # hand_obj_dist_thresh = 0.03


        a = 1.0
        b = 10.0
        c = 0.1
        d = 0.1


        if self._object_floor_contact() == True: #hand_obj_dist > hand_obj_dist_thresh:
            reward = - a * hand_obj_dist

        else:
            reward = - a * hand_obj_dist + b * obj_floor_dist - c * np.linalg.norm(obj_vel) + d * np.sum(self._hand_trq)
        
        return reward


    def _object_floor_contact(self):
        floorid = self.sim.model.geom_name2id('floor0')
        objectid = self.sim.model.geom_name2id('object0')
        ncon = self.sim.data.ncon
        contacts = self.sim.data.contact#[:ncon]

        for contact in contacts:
            if contact.geom1 == floorid:
                if contact.geom2 == objectid:
                    return True
            if contact.geom1 == objectid:
                if contact.geom2 == floorid:
                    return True
        return False