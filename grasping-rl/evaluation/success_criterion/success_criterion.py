from mujoco_py import load_model_from_path, MjSim, MjSimState
import numpy as np
import ipdb 

class SuccessCriterion:
	def __init__(self, mjsim):
		self.sim = mjsim
		self.model = self.sim.model

	def object_floor_contact(self):
		floorid = self.model.geom_name2id('floor0')
		objectid = self.model.geom_name2id('object0')
		ncon = self.sim.data.ncon
		contacts = self.sim.data.contact[:ncon]

		for contact in contacts:
			if contact.geom1 == floorid:
				if contact.geom2 == objectid:
					return True
			if contact.geom1 == objectid:
				if contact.geom2 == floorid:
					return True
		return False
	
	def velocity_criteria(self):
		obj_vel = self.sim.data.get_geom_xvelp('object0')                 
		if np.abs(obj_vel[2]) < 0.001:
			return True
		else:
			return False

	def grasp_criteria(self):
		if not self.object_floor_contact() and self.velocity_criteria():
			return 1
		else:
			return 0

