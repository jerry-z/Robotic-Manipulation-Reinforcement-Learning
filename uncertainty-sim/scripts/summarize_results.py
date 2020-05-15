#!/usr/bin/env
"""
Visualize results
"""
import sys
import os
import numpy as np
import ipdb
# for matplotlib 3D
import tkinter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def get_result_folders(experiment_name):
    data_root = os.environ['HAND_SIM_DATA']
    experiment_folder = os.path.join(data_root,experiment_name)
    results_folder = os.path.join(data_root,experiment_name,'results')
    return results_folder

def get_plot_fig_folder(experiment_name):
    data_root = os.environ['HAND_SIM_DATA']
    experiment_folder = os.path.join(data_root,experiment_name)
    plot_folder = os.path.join(data_root,experiment_name,'results_plot')
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)
    return plot_folder

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class PlotSummarizedResults():
    def __init__(self,experiment_name):
        results_folder = get_result_folders(sys.argv[1])
        self.experiment_name = sys.argv[1]
        self.pose_perturb = np.empty(shape=(0,6))
        self.grasp_score = np.array([])
        self.results_id = np.array([])
        for file_name in os.listdir(results_folder):
            self.pose_perturb = np.append(
                self.pose_perturb,
                np.array([np.load(os.path.join(results_folder,file_name,'perturb.npy'))]),
                axis=0)
            self.grasp_score = np.append(self.grasp_score,
                np.load(os.path.join(results_folder,file_name,'score.npy')))
            self.results_id = np.append(self.results_id,file_name)

    def plot_XY_map(self,z_select=0,display_id=False):
        # find all indices of z==z_select
        z_select = find_nearest(
            np.unique(self.pose_perturb[:,2]),z_select)
        indices = np.nonzero(self.pose_perturb[:,2]==z_select)[0]
        pose_perturb_Z = self.pose_perturb[indices,:]
        grasp_score_Z = self.grasp_score[indices]
        results_id_Z = self.results_id[indices]
        # need to generate a grid
        x_axis = np.unique(pose_perturb_Z[:,0])
        y_axis = np.unique(pose_perturb_Z[:,1])
        X_grid, Y_grid = np.meshgrid(x_axis,y_axis)
        score_grid = np.array(X_grid)
        results_id_grid = np.empty(shape=X_grid.shape)
        for x_loc in range(x_axis.size):
            for y_loc in range(y_axis.size):
                x = X_grid[y_loc,x_loc]
                y = Y_grid[y_loc,x_loc]
                index = np.nonzero(
                    (pose_perturb_Z[:,0]==x)&(pose_perturb_Z[:,1]==y))
                score_grid[y_loc,x_loc] = grasp_score_Z[index]
                results_id_grid[y_loc,x_loc] = results_id_Z[index]
        # plot the score map
        fig, (ax1) = plt.subplots()
        unit_change = 100.0
        score_color_cm = ax1.imshow(score_grid, cmap='Blues', interpolation='none')
        cbar = fig.colorbar(score_color_cm, ax=ax1)
        cbar.set_label('Grasping score')
        plt.xticks(
            np.arange(0,x_axis.size,x_axis.size-1), 
            x_axis[::x_axis.size-1]*unit_change)
        plt.xlabel('Disturbance in X [cm]')
        plt.yticks(
            np.arange(0,y_axis.size,y_axis.size-1), 
            y_axis[::y_axis.size-1]*unit_change)
        plt.ylabel('Disturbance in Y [cm]')
        # display experiment ID
        if display_id:
            for x_loc in range(x_axis.size):
                for y_loc in range(y_axis.size):
                    ax1.text(x_loc,y_loc,str(int(results_id_grid[y_loc,x_loc])))
        plt.title(f'Disturbance in Z = {z_select*unit_change} [cm]')
        # export figures
        z_select_str = str(z_select*unit_change)
        z_select_str = z_select_str.replace(".","_")
        z_select_str = z_select_str[:5] if len(z_select_str) > 5 else z_select_str
        fig.set_size_inches(10, 10)
        file_name = os.path.join(get_plot_fig_folder(self.experiment_name),f'Z_{z_select_str}_cm_ID_{str(display_id)}.png')
        plt.savefig(file_name,dpi=300)

if __name__ == '__main__':
    plot = PlotSummarizedResults(sys.argv[1])
    z_select_vals = np.linspace(-0.02,0.02,5)
    for z_select_val in z_select_vals:
        plot.plot_XY_map(z_select=z_select_val)
        plot.plot_XY_map(z_select=z_select_val,display_id=True)
        print(f'  Figures generated for [{sys.argv[1]}] z={z_select_val*100} cm')