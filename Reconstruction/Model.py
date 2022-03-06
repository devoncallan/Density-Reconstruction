#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:45:22 2022

@author: devoncallan
"""

import numpy as np
import math
PI = np.pi
SQRT3 = np.sqrt(3)
SQRT3_OVER_TWO = np.sqrt(3)/2
SQRT3_OVER_THREE = np.sqrt(3)/3
SQRT3_OVER_SIX = np.sqrt(3)/6

class Model:
    
    def __init__(self, name:str, f:tuple, p_dict:dict, 
                 xrange:tuple=None, yrange:tuple=None, step:float=None, 
                 a_nm:float = 1):
        
        self.name = name
    
        if xrange is None or yrange is None:
            # xrange = (-1, 1) 
            # yrange = (-SQRT3_OVER_TWO, SQRT3_OVER_TWO)
            xrange = (-SQRT3/2, SQRT3/2)
            yrange = (-1, 1) 
        if step is None:
            step = 0.02
        
        self.x_min, self.x_max = xrange
        self.y_min, self.y_max = yrange
        self.step = step
        self.x_list = self.__get_point_list(self.x_min, self.x_max, self.step)
        self.y_list = self.__get_point_list(self.y_min, self.y_max, self.step)
        
        self.a_nm = a_nm
        self.f = f
        
        if 'CS' in name:
            self.f_core, self.f_shell, self.f_matrix = f
            self.p_core = p_dict[self.f_core]
            self.p_shell = p_dict[self.f_shell]
            self.p_matrix = p_dict[self.f_matrix]
            
            self.r_core, self.r_shell = self.__calc_CS_dims()
            self.density = self.__make_CS_model()
            
        elif 'NL' in name:
            self.f_maj, self.f_min, self.f_matrix = f
            self.p_maj = p_dict[self.f_maj]
            self.p_min = p_dict[self.f_min]
            self.p_matrix = p_dict[self.f_matrix]
            
            self.r_maj, self.r_min = self.__calc_NL_dims()
            self.density = self.__make_NL_model()
        self.p_max = max(p_dict.values())
        self.p_min = min(p_dict.values())
        self.p_dict = p_dict
        self.p_avg = self.__get_p_avg()
        
    def __get_p_avg(self):
        '''Return average electron density of unit cell / model.'''
        return np.sum(np.multiply(np.asarray(list(self.p_dict.keys())),
                       np.asarray(list(self.p_dict.values()))))
    
    def __find_nearest(self, array:list, value:float):
        '''Returns index of closest value in array.'''
        array = np.asarray(array)
        idx = (np.abs(array-value)).argmin()
        return idx
    
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
    
    def __insert_tile(self, spot, tile):
        '''Copies each value of tile into spot if the value is not the 
        matrix. Returns the updated spot.'''
        
        for i in range(np.size(spot, 0)):
            for j in range(np.size(spot, 1)):
                val_spot = spot[i, j]
                val_tile = tile[i, j]
                if val_spot == self.p_matrix:
                    spot[i, j] = val_tile
        return spot
    
    def __calc_CS_dims(self):
        '''Calculate the dimension for a core-shell model given volume
        fractions for the core and shell cylinders. Returns radius of core and 
        shell cylinders.'''
        area_unit_cell = (self.a_nm ** 2) * np.cos(PI/6)
        area_core_shell = (self.f_core + self.f_shell) * area_unit_cell
        area_core = (self.f_core / (self.f_core + self.f_shell)) * area_core_shell
        
        r_shell_cyl = np.sqrt(area_core_shell / PI)
        r_core_cyl  = np.sqrt(area_core / PI)
        
        return r_core_cyl, r_shell_cyl
    
    def __calc_NL_dims(self):
        '''Calculate the dimension for a nested latticee model given volume 
        fractions for the major and minor cylinders. Returns radius of 
        major and minor cylinders.'''
        area_unit_cell = (self.a_nm ** 2) * np.cos(PI/6)
        area_maj_cyl = self.f_maj * area_unit_cell
        area_min_cyl = self.f_min * area_unit_cell / 2
        
        r_maj_cyl = np.sqrt(area_maj_cyl / PI)
        r_min_cyl = np.sqrt(area_min_cyl / PI)
        
        return r_maj_cyl, r_min_cyl
    
    def __make_CS_tile(self):
        '''Generates 2D matrix representation of a core-shell cylinder tile 
        given volume fractions and electron densities for the core, shell, 
        and matrix.'''
        width_tile = int(round(2 * self.r_shell / self.step)) + 1
        x_origin_tile, y_origin_tile = width_tile/2, width_tile/2
        tile = self.p_matrix * np.ones((width_tile, width_tile))
        get_dist = lambda i, j: math.sqrt((i - y_origin_tile)**2
                                          + (j - x_origin_tile)**2)
        
        for i in range(width_tile):
            for j in range(width_tile):
                r = self.step * get_dist(i, j)
                
                if r <= self.r_core:
                    tile[i, j] = self.p_core
                elif r <= self.r_shell:
                    tile[i, j] = self.p_shell      
        return tile
    
    def __make_NL_tile(self, maj_cyl: bool):
        '''Generates 2D matrix representation of a cylinder given '''   
        if maj_cyl:
            r_cyl = self.r_maj
            p_cyl = self.p_maj
        else:
            r_cyl = self.r_min
            p_cyl = self.p_min
        
        width_tile = int(round(2 * r_cyl / self.step)) + 1
        x_origin_tile, y_origin_tile = width_tile/2, width_tile/2
        tile = self.p_matrix * np.ones((width_tile, width_tile))
        get_dist = lambda i, j: math.sqrt((i - y_origin_tile)**2
                                          + (j - x_origin_tile)**2)
        
        for i in range(width_tile):
            for j in range(width_tile):
                r = self.step * get_dist(i, j)
                
                if r <= r_cyl:
                    tile[i, j] = p_cyl 
        return tile
        
    def __make_CS_model(self):
        '''Generates 2D matrix representation of a core-shell cylinder model
        given volume fractions and electron densities for the core, shell,
        and matrix.'''
        # Create model box points that is double the size of (x_list, y_list)
        # to fully insert cylinder tiles before cropping.
        x_list_big = self.__get_point_list(2*self.x_list[0], 
                                         2*self.x_list[len(self.x_list)-1], 
                                         self.step)
        y_list_big = self.__get_point_list(2*self.y_list[0],
                                         2*self.y_list[len(self.y_list)-1], 
                                         self.step)
        model_big = self.p_matrix * np.ones((len(y_list_big), len(x_list_big)))
        
        tile = self.__make_CS_tile()
        width_tile = int(len(tile[0]))
        
        # Fractional x, y coordinates of cylinders in model
            
        # coords_cyl = [(0, 0), (-1, 0), (1, 0), 
        #               (-0.5, SQRT3/2), (0.5, SQRT3/2),
        #               (-0.5, -SQRT3/2), (0.5, -SQRT3/2)]
        coords_cyl = [(0, 0), (0, -1), (0, 1),
                      (SQRT3/2, -0.5), (SQRT3/2, 0.5),
                      (-SQRT3/2, -0.5), (-SQRT3/2, 0.5)]
        coords_cyl = np.multiply(coords_cyl, self.a_nm)
        
        # Insert each cylinder tile into model
        for coord in coords_cyl:
            x_origin = self.__find_nearest(x_list_big, coord[0])
            y_origin = self.__find_nearest(y_list_big, coord[1])
            
            # Define indices of bounding box for inserting tile
            x_min = int(round(x_origin - width_tile/2))
            y_min = int(round(y_origin - width_tile/2))
            x_max = int(round(x_min + width_tile))
            y_max = int(round(y_min + width_tile))
            spot = model_big[y_min : y_max, x_min : x_max]
            
            model_big[y_min:y_max, x_min : x_max] = self.__insert_tile(spot, tile)
            
        # Crops model box to be original size of self.x_list, self.y_lis
        width_x_model = len(self.x_list)    
        width_y_model = len(self.y_list)
        x_min_model = self.__find_nearest(x_list_big, self.x_list[0])
        y_min_model = self.__find_nearest(y_list_big, self.y_list[0])
        model = model_big[y_min_model : y_min_model+width_y_model,
                          x_min_model : x_min_model+width_x_model]
        return model
        
    def __make_NL_model(self):
        '''Generates 2D matrix representation of a core-shell cylinder model
        given volume fractions and electron densities for the core, shell,
        and matrix.'''
        # Create model box points that is double the size of (x_list, y_list)
        # to fully insert cylinder tiles before cropping.
        x_list_big = self.__get_point_list(2*self.x_list[0], 
                                         2*self.x_list[len(self.x_list)-1], 
                                         self.step)
        y_list_big = self.__get_point_list(2*self.y_list[0],
                                         2*self.y_list[len(self.y_list)-1], 
                                         self.step)
        model_big = self.p_matrix * np.ones((len(y_list_big), len(x_list_big)))
        
        tile_maj = self.__make_NL_tile(maj_cyl=True)
        tile_min = self.__make_NL_tile(maj_cyl=False)
        width_tile_maj = int(len(tile_maj[0]))
        width_tile_min = int(len(tile_min[0]))
        
        # Fractional x, y coordinates of cylinders in model
        # coords_maj = [(0, 0), (-1, 0), (1, 0), 
        #               (-0.5, SQRT3/2), (0.5, SQRT3/2),
        #               (-0.5, -SQRT3/2), (0.5, -SQRT3/2)]
        coords_maj = [(0, 0), (0, -1), (0, 1),
                      (SQRT3/2, -0.5), (SQRT3/2, 0.5),
                      (-SQRT3/2, -0.5), (-SQRT3/2, 0.5)]
        
        # coords_min = [(0, SQRT3/3), (0, -SQRT3/3),
        #               (-0.5, SQRT3/6), (0.5, SQRT3/6),
        #               (-0.5, -SQRT3/6), (0.5, -SQRT3/6),
        #               (-1, SQRT3/3), (1, SQRT3/3),
        #               (-1, -SQRT3/3), (1, -SQRT3/3)]
        coords_min = [(SQRT3/3, 0), (-SQRT3/3, 0),
                      (SQRT3/6, -0.5), (SQRT3/6, 0.5),
                      (-SQRT3/6, -0.5), (-SQRT3/6, 0.5),
                      (SQRT3/3, -1), (SQRT3/3, 1),
                      (-SQRT3/3, -1), (-SQRT3/3, 1)]
        
        coords_maj = np.multiply(coords_maj, self.a_nm)
        coords_min = np.multiply(coords_min, self.a_nm)
        coords = [coords_maj, coords_min]
        
        for coord_type, coord_list in enumerate(coords):
            for coord in coord_list:
                
                if coord_type == 0:
                    tile = tile_maj
                    width_tile = width_tile_maj
                elif coord_type == 1:
                    tile = tile_min
                    width_tile = width_tile_min
                
                x_origin = self.__find_nearest(x_list_big, coord[0])
                y_origin = self.__find_nearest(y_list_big, coord[1])
                
                # Define indices of bounding box for inserting tile
                x_min = int(round(x_origin - width_tile/2))
                y_min = int(round(y_origin - width_tile/2))
                x_max = int(round(x_min + width_tile))
                y_max = int(round(y_min + width_tile))
                spot = model_big[y_min : y_max, x_min : x_max]
                
                model_big[y_min:y_max, x_min:x_max] = self.__insert_tile(spot, tile)
                
        # Crops model box to be original size of self.x_list, self.y_list
        width_x_model = len(self.x_list)    
        width_y_model = len(self.y_list)
        x_min_model = self.__find_nearest(x_list_big, self.x_list[0])
        y_min_model = self.__find_nearest(y_list_big, self.y_list[0])
        model = model_big[y_min_model : y_min_model+width_y_model,
                          x_min_model : x_min_model+width_x_model]
        return model
    
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
                            