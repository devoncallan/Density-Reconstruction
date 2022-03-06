#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 02:19:52 2022

@author: devoncallan
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
SQRT3 = np.sqrt(3)

def plot_1D_density(recon=None, model=None, x_val:float=None, y_val:float=None, title='', dpi=150, figsize=(6, 6), ylim=None):
    '''Plots a slice of the model and reconstruction at a constant x or y value.'''
    if recon is None and model is None:
        raise Exception('Pass in Reconstruction2D or Model object.')
    elif x_val is None and y_val is None:
        raise Exception('Pass in valid x_val or y_val.')
        
    plt.figure(figsize=figsize, dpi=dpi)
    legend = []
    fontsize=14
    if not model is None:
        if not x_val is None:
            y_list_model, density_model_ylc = model.get_slice(x_val=x_val)
            plt.plot(y_list_model, density_model_ylc, 'k--')
            plt.xlabel('y / a', fontsize=fontsize)
        elif not y_val is None:
            x_list_model, density_model_xlc = model.get_slice(y_val=y_val)
            plt.plot(x_list_model, density_model_xlc, 'k--')
            plt.xlabel('x / a', fontsize=fontsize)
        plt.yticks(list(model.p_dict.values()), fontsize=fontsize)
        legend.append(model.name)
    if not recon is None:
        if not x_val is None:
            y_list_recon, density_recon_ylc = recon.get_slice(x_val=x_val)
            plt.plot(y_list_recon, density_recon_ylc, color='tab:blue')
            plt.xlabel('y / a', fontsize=fontsize)
        elif not y_val is None:
            x_list_recon, density_recon_xlc = recon.get_slice(y_val=y_val)
            plt.plot(x_list_recon, density_recon_xlc, color='tab:blue')
            plt.xlabel('x / a', fontsize=fontsize)
        legend.append(recon.name) 
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=fontsize)
    plt.legend(legend, frameon=False)
    plt.ylabel('Electron density (e$^{-}$/ Å$^{3}$)', fontsize=fontsize)
    if ylim is None:
        plt.ylim([0.3, 0.48])
    plt.ylim(ylim)
    plt.show()

def plot_2D_density(model, title='', dpi=150, figsize=(3, 4), cmap=cm.get_cmap('viridis')):
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(model.name, fontsize='24')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xticks(np.arange(-1, 1.1, 0.5))
    # plt.yticks(np.arange(-1, 1.1, 0.5))
    # extent=[-1, 1, -SQRT3/2, SQRT3/2]
    extent = [model.x_min, model.x_max, model.y_min, model.y_max]
    plt.imshow(model.density, extent=extent,
               cmap=cmap, origin='lower')
    # cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Relative Electron Density (a.u.)')
    plt.show()
    
def plot_R2_histogram(sorted_dict, N:int=None, highlight_model:str=None, ylabel='', title='', dpi=150, figsize=(6, 6)):
       
    if N is None or N > len(sorted_dict):
        N = len(sorted_dict)
    plt.figure(figsize=figsize, dpi=dpi)
    
    x_hist = np.arange(0, N, 1)
    hist_data = [data[1] for data in sorted_dict]
    hist_data = hist_data[0:N]
    barlist = plt.bar(x_hist, hist_data, width=0.6)
    
    if not highlight_model is None:
        for i, (model_str, data) in enumerate(sorted_dict):
            
            if highlight_model in model_str:
                barlist[i].set_color('tab:blue')
            else:
                barlist[i].set_color('tab:gray')
            if i == N-1:
                break
        plt.legend([highlight_model], frameon=False, fontsize=12, loc='upper right')
    
    plt.title(title, fontsize='24')
    plt.xlabel('Model-Reconstruction Pair (Descending $R^2$)', labelpad=10, fontsize=14)
    plt.ylabel('$R^2$', fontsize=14)
    plt.xticks([])
    plt.ylim([0, 1])
    plt.xlim([-1, N])
    plt.show()
    
def plot_3D_density(model, title='', x_val:float=None, y_val:float=None, dpi=150, figsize=(3, 4), cmap=cm.get_cmap('viridis')):
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ha = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(model.x_list, model.y_list, indexing='xy')
    
    density = gaussian_filter(model.density, 2)
    if not x_val is None:
        y_list, density_ylc = model.get_slice(x_val=x_val)
        x_list = np.multiply(np.ones_like(y_list), x_val)
        ha.plot3D(x_list, y_list, density_ylc, '-', color='k', linewidth=0.75, zorder=10)
    elif not y_val is None:
        x_list, density_xlc = model.get_slice(y_val=y_val)
        y_list = np.multiply(np.ones_like(x_list), y_val)
        ha.plot3D(x_list, y_list, density_xlc, '-', color='k', linewidth=0.75, zorder=10)
    
    # ha.plot3D(x_ylc + 1, y_ylc, ylc_model, '--', color='black', linewidth=0.75, zorder=10)
    s = ha.plot_surface(X, Y, density, cmap=cmap, linewidth=0, rcount=10000, ccount=10000, antialiased=False)
    
    ha.set_frame_on(False)
    ha.set_zlim([0.3, 0.6])
    ha.set_xlim([-1, 1])
    ha.set_ylim([-1, 1])
    ha.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ha.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ha.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ha.set_axis_off()
    cbar = fig.colorbar(s, aspect=5, fraction=0.04, pad=0.005)
    cbar.set_ticks([])
    
    plt.show()

def plot_all_models():
    return None