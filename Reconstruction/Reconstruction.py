#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 00:42:08 2022

@author: devoncallan
"""
import numpy as np
PI = np.pi
SQRT3 = np.sqrt(3)
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from copy import deepcopy

class ReconstructionSet:
    
    def __init__(self, F_coeff:list=None, 
                 num_peaks:int=None, xrange:tuple=None, 
                 yrange:tuple=None, step:float=None, a_nm:float=1):
        
        if xrange is None or yrange is None:
            xrange = (-SQRT3/2, SQRT3/2)
            yrange = (-1, 1) 
        if step is None:
            step = 0.02
        if num_peaks is None:
            num_peaks = len(F_coeff)
            
        self.x_min, self.x_max = xrange
        self.y_min, self.y_max = yrange
        self.step = step
        
        self.x_list = self.__get_point_list(self.x_min, self.x_max, self.step)
        self.y_list = self.__get_point_list(self.y_min, self.y_max, self.step)

        self.F_coeff = F_coeff
        self.num_peaks = num_peaks
        self.a_nm = a_nm
        self.hk_list = self.get_hk_list()
        
        self.reconstructions = {}
        self.densities = {}
        
        
    def get_args(self):
        args = {}
        args['F_coeff'] = self.F_coeff
        args['x_list'] = self.x_list
        args['y_list'] = self.y_list
        args['hk_list'] = self.hk_list
        args['a_nm'] = self.a_nm
        args['step'] = self.step
        return args
    
    def __get_point_list(self, min_val:float, max_val:float, step:float):
        '''Returns array of numbers evenly spaced by step from min_val to 
        max_val. min_val and max_val are rounded to precision of step such
        that each point is a multiple of step.'''
        def round_to(value: float, precision: float):
            '''Returns value rounded to given precision.'''
            return round(value / precision) * precision
        
        min_val = round_to(min_val, step)
        max_val = round_to(max_val, step)
        num = int(round((max_val - min_val)/step)) + 1
        return np.linspace(min_val, max_val, num)

    # GET ALL COMBINATIONS OF PHASES (-1 and 1) FOR GIVEN LENGTH
    def enumerate_phases(self, num_peaks:int=None, inv_phases:bool=False):
        '''Lists all possible combinations of phases (-1 and 1) for a given 
        number of peaks.'''
        if num_peaks is None:
            num_peaks = self.num_peaks
            num_phases = 2 ** (self.num_peaks - int(not inv_phases))
        else:
             num_phases = 2 ** (num_peaks - int(not inv_phases))
            
        phase_list = ['{:0{}b}'.format(i, num_peaks) for i in range(num_phases)]
        for i, str_phase in enumerate(phase_list):
            phase = np.zeros(len(str_phase))
            for j, char in enumerate(str_phase):
                if char == '0':
                    phase[j] = 1
                elif char == '1':
                    phase[j] = -1
            phase_list[i] = list(phase)
        return phase_list
    
    def get_hk_list(self):
        hk_10 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        hk_11 = [(1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2)]
        hk_20 = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2)]
        hk_21 = [(2, 1), (-2, -1), (1, 2), (-1, -2), (3, -1), (-3, 1), 
               (1, -3), (-1, 3), (3, -2), (-3, 2), (2, -3), (-2, 3)]
        hk_30 = [(3, 0), (-3, 0), (0, 3), (0, -3), (3, -3), (-3, 3)]
        hk_22 = [(2, 2), (-2, -2), (4, -2), (-4, 2), (2, -4), (-2, 4)]
        hk_31 = [(3, 1), (-3, -1), (1, 3), (-1, -3), (4, -1), (-4, 1), 
                (1, -4), (-1, 4), (4, -3), (-4, 3), (3, -4), (-3, 4)]
        hk_40 = [(4, 0), (-4, 0), (0, 4), (0, -4), (4, -4), (-4, 4)]
        hk_32 = [(3, 2), (-3, -2), (2, 3), (-2, -3), (5, -2), (-5, 2), 
                 (2, -5), (-2, 5), (5, -3), (-5, 3), (3, -5), (-3, 5)]
        hk_41 = [(4, 1), (-4, -1), (1, 4), (-1, -4), (5, -1), (-5, 1), 
                 (1, -5), (-1, 5), (5, -4), (-5, 4), (4, -5), (-4, 5)]
        hk_50 = [(5, 0), (-5, 0), (0, 5), (0, -5), (5, -5), (-5, 5)]
        hk_33 = [(3, 3), (-3, -3), (6, -3), (-6, 3), (3, -6), (-3, 6)]
        hk_42 = [(4, 2), (-4, -2), (2, 4), (-2, -4), (6, -2), (-6, 2), 
                 (2, -6), (-2, 6), (6, -4), (-6, 4), (4, -6), (-4, 6)]
        hk_51 = [(5, 1), (-5, -1), (1, 5), (-1, -5), (6, -1), (-6, 1), 
                 (1, -6), (-1, 6), (6, -5), (-6, 5), (5, -6), (-5, 6)]
        hk_60 = [(6, 0), (-6, 0), (0, 6), (0, -6), (6, -6), (-6, 6)]
        hk_all = [hk_10, hk_11, hk_20, hk_21, hk_30, hk_22, hk_31, hk_40,
                  hk_32, hk_41, hk_50, hk_33, hk_42, hk_51, hk_60]
        hk = hk_all[0:self.num_peaks]
        hk_cart = hk # initialize hk_cart with same dimensions as hk

        # CONVERT h,k FROM HEXAGONAL TO CARTESIAN COORDINATES
        for m, hk_list in enumerate(hk):
            for n, (h, k) in enumerate(hk_list):
                
                h_cart = h + k*np.cos(PI/3) 
                k_cart = k*np.sin(PI/3)
                 
                hk_cart[m][n] = (h_cart, k_cart)
        return hk_cart
    
    def add_reconstruction(self, reconstruction):
        phase_str = reconstruction.phase_str
        self.reconstructions[phase_str] = reconstruction
        self.densities[phase_str] = reconstruction.density
        
    def normalize_all(self):
        max_val = np.amax(np.abs(np.concatenate(list(self.densities.values()))))
        for phase_str in self.reconstructions:
            recon = self.reconstructions[phase_str]
            recon.density = np.divide(recon.density, max_val)
            self.add_reconstruction(recon)
    
class Reconstruction:
    
    def __init__(self, F_coeff:list, x_list:list, y_list:list, hk_list:list,
                 phase:list=None, phase_str:str=None, a_nm:float=1, step:float=None):
    
        if phase_str is None and not phase is None:
            phase_str = self.get_phase_str(phase)
        elif phase is None and not phase_str is None:
            phase = self.get_phase_list(phase_str)
        else:
            raise TypeError('Must include phase or phase_str.')
            
        self.num_peaks = len(phase)
        self.F_coeff = F_coeff[0:self.num_peaks]
        self.hk_list = hk_list[0:self.num_peaks]
        
        self.x_list = x_list
        self.y_list = y_list
        self.step = step
        
        self.x_min = min(x_list)
        self.x_max = max(x_list)
        self.y_min = min(y_list)
        self.y_max = max(y_list)
        
        self.phase = phase
        self.phase_str = phase_str
        self.name = phase_str

        self.a_nm = a_nm
        self.density = None
        self.density_ref = None
        self.model_fits = {}

    def get_phase_str(self, phase_list:list):
        '''Interprets 1 as '+' and -1 as '-'. Returns combined string.'''
        phase_str = ''
        for phase in phase_list:
            if phase == 1:
                phase_str += '+'
            elif phase == -1:
                phase_str += '-'
        return phase_str

    def __find_nearest(self, array:list, value:float):
        '''Returns index of closest value in array.'''
        array = np.asarray(array)
        idx = (np.abs(array-value)).argmin()
        return idx

    def get_phase_list(self, phase_str:str):
        '''Interprets '+' as 1 and '-' as -1. Returns list of phases.'''
        phase_char_list = list(phase_str)
        phase_list = np.zeros(len(phase_char_list))
        for i, phase_char in enumerate(phase_char_list):
            if phase_str == '+':
                phase_list[i] = 1
            elif phase_str == '-':
                phase_list[i] = -1
        return phase_list
        
    def calculate_density(self, normalize=True):
        '''Calculate the reconstructed electron density.'''
        X, Y = np.meshgrid(self.x_list, self.y_list, indexing='xy')
        density = np.zeros((len(self.y_list), len(self.x_list)))
        
        
        scale = 1 / (self.a_nm * SQRT3/2)
        
        for n, hk in enumerate(self.hk_list):
            for h, k in hk:
                F = np.cos(2 * PI * scale * (h*X + k*Y))
                density += self.F_coeff[n]*self.phase[n]*F
        
        if normalize == True:
            density = np.divide(density, np.amax(np.abs(density)))
        self.density = density
        return density
    
    def get_slice(self, x_val:float=None, y_val:float=None):
        '''Gets a slice of the model at a constant x or y value.
        Returns an array of the x and y values of the slice.'''
        if not x_val is None:
            x_idx = self.__find_nearest(self.x_list, x_val)
            if np.abs(self.x_list[x_idx] - x_val) > self.step:
                raise Exception('Value outside range of model.')
            return self.y_list, self.density[:, x_idx]
        elif not y_val is None:
            y_idx = self.__find_nearest(self.y_list, y_val)
            if np.abs(self.y_list[y_idx] - y_val) > self.step:
                raise Exception('Value outside range of model.')
            return self.x_list, self.density[y_idx, :]
        
    def __new_reconstruction(self, p_fit):
        '''Returns copy of current reconstruction with updated density.'''
        new_recon = deepcopy(self)
        new_recon.density = p_fit
        return new_recon
        
    def fit_to_model(self, model):
        '''Fits the reconstruction to a given electron density model.
        p_fit = <p> + K * p_reconstruction. Average electron density is given
        by the model. '''
        
        def __scale(p_fit, K):
            return np.multiply(p_fit, K) + model.p_avg
        
        def __error_parameters(p_model, residuals):
            '''SSR: Residual sum of squares
               SST: Total sum of squares
               R2 : R^2 (variance something. '''
            # error_params = {}
            # error_params['SSR'] = np.sum(residuals ** 2)
            # error_params['SST'] = np.sum((p_model - np.mean(p_model)) ** 2)
            # error_params['R2']  = 1 - error_params['SSR'] / error_params['SST']
            ssr = np.sum(residuals ** 2)
            sst = np.sum((p_model - np.mean(p_model)) ** 2)
            r_squared  = 1 - ssr / sst
            return (ssr, r_squared)
            
        # Define model density 
        p_model = model.density
        
        # Normalize by maximium magnitude of reconstruction density
        p_recon = self.density
        p_fit = np.divide(p_recon, np.amax(np.abs(p_recon)))
        
        # Checks if model and reconstruction dimensions are equal
        if np.shape(p_model) != np.shape(p_fit):
            raise Exception('Model and Reconstruction must be the same size.')

        # Fit the reconstruction to a model
        # popt is the fit parameter, K; pcov is the covariance matrix
        popt, pcov = curve_fit(__scale, np.ravel(p_fit), np.ravel(p_model), bounds=(0, np.inf))
        
        # Get fitted electron density map and generates new reconstruction
        p_fit = __scale(p_fit, *popt)
        new_recon = self.__new_reconstruction(p_fit)
        
        # Defines error parameters
        residuals = p_model - p_fit
        error_params = __error_parameters(p_model, residuals)
        
        return new_recon, popt, error_params
        