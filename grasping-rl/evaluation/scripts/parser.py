#!/usr/bin/env python3

import numpy as np
import ipdb

class BaselineParser:
	def __init__(self, csv_file):
		self.base = csv_file
	def get_testcase(self,testcase):
		i  = testcase
		initial_obj_com = np.array((self.base['x'][i],self.base['y'][i],0))

		params = {}

		if self.base['shape'][i] == 'Cube':

			params['object'] = 'Cube'
			params['size'] = np.array((self.base['len'][i], self.base['width'][i], self.base['height'][i]))	
			params['mass'] = 0.75#self.base['len'][i]*self.base['width'][i]*self.base['height'][i]*1000
			params['pos'] = np.array((self.base['x'][i],self.base['y'][i],0.25))
			params['quat'] = euler2quat([0,0,self.base['rot_z'][i]])		
			params['type'] = 6
			ixx = 1/12 * params['mass'] * (self.base['height'][i]**2 + self.base['width'][i]**2)
			iyy = 1/12 * params['mass'] * (self.base['len'][i]**2 + self.base['height'][i]**2)
			izz = 1/12 * params['mass'] * (self.base['len'][i]**2 + self.base['width'][i]**2)
			params['inertia'] = ixx,iyy,izz

		if self.base['shape'][i] == 'Sphere':

			params['object'] = 'Sphere'
			params['size'] = np.array((self.base['radius'][i],0,0))
			params['mass'] = 0.75#4/3*3.14*(self.base['radius'][i]**3)*1000
			params['pos'] = np.array((self.base['x'][i],self.base['y'][i],0.25))
			params['quat'] = euler2quat([0,0,self.base['rot_z'][i]])		
			params['type'] = 2
			ixx = iyy = izz = 2/5*(self.base['radius'][i]**2)*params['mass']	
			params['inertia'] = ixx,iyy,izz

		elif self.base['shape'][i] == 'Cyl':

			params['object'] = 'Cyl'
			params['mass'] = 0.75#3.14*(self.base['radius'][i]**2)*self.base['height'][i]*1000
			params['pos'] = np.array((self.base['x'][i],self.base['y'][i],0.25))
			params['size'] = np.array((self.base['radius'][i], self.base['height'][i], 0))
			params['quat'] = euler2quat([0,0,0])		
			params['type'] = 5
			ixx = iyy = 1/12 *(3 * self.base['radius'][i]**2 + self.base['height'][i]**2) * params['mass']
			izz = 1/12*(self.base['radius'][i]**2)*params['mass']
			params['inertia'] = ixx,iyy,izz

		return params

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
