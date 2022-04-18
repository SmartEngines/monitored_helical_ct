#!/usr/bin/env python
# coding: utf-8

import astra
import numpy as np
import pandas as pd
import io
import math
import os
from timeit import default_timer as tmr
from random import shuffle
from PIL import Image
from cv2 import imread, IMREAD_UNCHANGED
import sys

import keras.backend as k
from keras_retinanet import layers
import keras_retinanet
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Flatten, Dense,Input

def print_to_file(*args):
    s = ' '.join([str(x) for x in args])
    print(s)
    with open(path_to_log, 'a') as f:
        f.write(s + "\n")

class Patient:
    
    '''
    Patient class to store the metadata read from Patients_metadata.csv file
    Data is read by a method load_images()
    '''
    
    data_folder = ""
    
    def __init__(self, full_name):
        self.is_ok = True
        self.full_name = full_name
        a = full_name.split("/")
        self.SR = a[-1]
        self.id = a[-2]
        patient_data = pd.read_csv(Patient.data_folder + "CSV/Patients_metadata.csv")
        patient_data.set_index("Patient ID", inplace = True)
        table_height = patient_data[self.SR + " TableHeight"][self.id]
        self.source_det  = patient_data[self.SR + " DistanceSourceToDetector"][self.id]
        self.source_obj = patient_data[self.SR + " DistanceSourceToPatient"][self.id] + table_height
        self.slice_thickness = patient_data[self.SR + " SliceThickness"][self.id]
        self.pitch = patient_data[self.SR + " SpiralPitchFactor"][self.id]
        self.table_move_per_rotation = patient_data[self.SR + " TableFeedPerRotation"][self.id]
        pixel_spacing = patient_data[self.SR + " PixelSpacing"][self.id]
        self.markup = a[-3]
        if pd.isna(self.slice_thickness):
            print_to_file("bad patient " + self.id + " " + self.SR + " nan slice thickness")
            self.is_ok = False
            return
        if pd.isna(pixel_spacing):
            print_to_file("bad patient " + self.id + " " + self.SR + " nan pixel spacing")
            self.is_ok = False
            return
        self.vol_px_size = float(pixel_spacing.split(", ")[0][1:])
        df = Patient.data_folder + self.full_name + "/"
        self.size_z = len(os.listdir(df))
            
    def load_images(self):
        df = Patient.data_folder + self.full_name + "/"
        self.data = np.zeros((self.size_z,512,512))
        for i, d in enumerate(sorted(os.listdir(df))):
            self.data[i,:,:] = imread(df + str(d), IMREAD_UNCHANGED)

class Experiment:
    
    '''Experiment class with main method run()'''
    
    def __init__(self, full_name, slice_to_rec, reproection_save_path, predictions_save_path, NN_path, skip_type, stopping_interval):
        self.full_name = full_name
        self.slice_to_rec = slice_to_rec
        self.reproection_save_path = reproection_save_path
        self.predictions_save_path = predictions_save_path
        self.NN_path = NN_path
        self.skip_type = skip_type
        self.stopping_interval = stopping_interval
        self.patient = Patient(full_name)
        self.predictions = []
        with open(predictions_save_path, 'w') as f:
            f.write('')
        self.size_z = self.patient.size_z
        self.size_x = 512
        self.size_y = 512
        self.det_size_x = 736
        self.det_size_y = 16
        self.source_obj = self.patient.source_obj
        self.obj_det = self.patient.source_det - self.source_obj
        self.det_spacing_x = 1.2
        self.det_spacing_y = 1.2
        self.pitch = self.patient.pitch
        self.voxel_size = self.patient.vol_px_size
        self.slice_thickness = self.patient.slice_thickness
        self.angles_per_rotation = 600
        self.source_obj    /= self.voxel_size
        self.obj_det       /= self.voxel_size
        self.det_spacing_x /= self.voxel_size
        self.det_spacing_y /= self.slice_thickness
        self.pitch_dist = self.pitch * self.det_size_y *  self.det_spacing_y
        self.start_scan_position = - self.det_size_y * self.det_spacing_y

                    
    def predict_and_save_rec(self, rec, proj_count, sector_count, start_z):
        '''
        making and saving a prediction for the current layer
        saving a reconstruction for the current layer
        '''
        size_z = rec.shape[0]
        h = rec.shape[1]
        w = rec.shape[2]
        pc = f'{sector_count:04}'
        path = f'{self.reproection_save_path}{self.full_name}/{(self.slice_to_rec+1):04}/'
        try: 
            os.makedirs(path)
        except OSError as error:
            pass
        
        # load NN model and make prediction for the current layer
        k.clear_session()
        custom_object={'UpsampleLike': keras_retinanet.layers._misc.UpsampleLike}
        net = load_model(self.NN_path, custom_objects=custom_object)
        i = self.slice_to_rec - start_z
        pred_ind = net.predict(np.expand_dims(np.expand_dims(rec[i,:,:],axis=0),axis=3))[0][0]
        
        # save prediction
        self.predictions.append((proj_count, pred_ind))
        with open(self.predictions_save_path, 'a') as f:
            f.write(str(proj_count) + ' ' + str(pred_ind) + '\n')
            
        # save reconstruction
        im = Image.fromarray(rec[i,:,:].astype(np.float32))
        im.convert("L")
        im.save(f"{path}IM{pc}.tif")
        k.clear_session()
    
    def get_min_max_prj(self, idx):
        '''
        Return min and max indexes of projections 
        influencing the reconstruction of the current layer
        '''
        z_min = idx
        z_max = idx+1
        min_dist = -self.start_scan_position + z_min - self.det_size_y * self.det_spacing_y / 2
        max_dist = -self.start_scan_position + z_max + self.det_size_y * self.det_spacing_y / 2
        i_min = min_dist / self.pitch_dist * self.angles_per_rotation
        i_max = max_dist / self.pitch_dist * self.angles_per_rotation
        return (int(i_min), int(i_max))

    def run(self):
        '''main method'''
        if not self.patient.is_ok:
            print_to_file("bad patient " + p.id + " " + p.SR)
            return
        
        self.patient.load_images()
        
        max_rec_size = 7 # max z size for the reconstruction to limit the volume
        trust_count = 4  
        
        # calculating the limits of projections needed for reconstruction
        slice_min_prj, slice_max_prj = self.get_min_max_prj(self.slice_to_rec)
        slice_max_prj += (stopping_interval - slice_max_prj % stopping_interval)
        slice_min_prj += (stopping_interval - slice_min_prj % stopping_interval) - stopping_interval
        max_sectors_count = (slice_max_prj - slice_min_prj) // stopping_interval
        
        sectors = list(range(max_sectors_count))
        shuffle(sectors)
        print_to_file("shuffled sectors:", sectors)
        
        # precalculating full projection data
        pre_calc_vec_geom = np.zeros((slice_max_prj, 12))
        for i in range(slice_max_prj):
            angle = 2 * 3.1415926535*i/self.angles_per_rotation
            z = self.start_scan_position - self.size_z / 2 + (i/self.angles_per_rotation) * self.pitch_dist
            pre_calc_vec_geom[i, 0] =  math.sin(angle)*self.source_obj
            pre_calc_vec_geom[i, 1] = -math.cos(angle)*self.source_obj
            pre_calc_vec_geom[i, 2] = z

            pre_calc_vec_geom[i, 3] = -math.sin(angle)*self.obj_det
            pre_calc_vec_geom[i, 4] =  math.cos(angle)*self.obj_det
            pre_calc_vec_geom[i, 5] = z

            pre_calc_vec_geom[i, 6] = math.cos(angle)*self.det_spacing_x
            pre_calc_vec_geom[i, 7] = math.sin(angle)*self.det_spacing_x
            pre_calc_vec_geom[i, 8] = 0

            pre_calc_vec_geom[i, 9] = 0
            pre_calc_vec_geom[i, 10] = 0
            pre_calc_vec_geom[i, 11] = self.det_spacing_y

        pre_calc_vol_geom = astra.create_vol_geom(self.size_y, self.size_x, self.size_z)
        pre_calc_proj_geom = astra.create_proj_geom('cone_vec', self.det_size_y, self.det_size_x, pre_calc_vec_geom)
        pre_calc_proj_id, pre_calc_proj_data = astra.create_sino3d_gpu(self.patient.data, pre_calc_proj_geom, pre_calc_vol_geom)

        # main loop for sectors activating
        for cur_sectors_count in range(len(sectors)):
            print(f"curent_progress:{cur_sectors_count}/{len(sectors)}", end = '\r')
            
            # set activation types for the sectors and put the necessary idxs to a container
            if skip_type == "partial":
                sectors_type = np.ones(max_sectors_count)
            else:
                sectors_type = np.zeros(max_sectors_count)
            if skip_type == "full":
                inc = 2
            else:
                inc = 1
            for i in sectors[:cur_sectors_count+1]:
                sectors_type[i] += inc
            sectors_to_add = sectors[:cur_sectors_count+1]
            container = list(range(slice_min_prj))
            for i in range(max_sectors_count):
                if sectors_type[i] == 1:
                    container.append(slice_min_prj + i*10 + 4)
                    container.append(slice_min_prj + i*10 + 7)
                elif sectors_type[i] == 2:
                    for j in range(10):
                        container.append(slice_min_prj + i*10 + j)
            
            # chose z axis limits for reconstruction
            rec_stop = int(((slice_max_prj / self.angles_per_rotation) * self.pitch_dist) + self.start_scan_position + self.det_size_y *  self.det_spacing_y / 2)
            rec_stop = min(rec_stop, self.size_z)
            rec_start = max(rec_stop - max_rec_size, 0)
            total_min_prj = int(max(rec_start/(self.pitch_dist) * self.angles_per_rotation, 0))
            rec_size_z = rec_stop - rec_start
            
            # set new geometry consisted with a precalculated one and with a chosen limits 
            tmp_vec_geom = np.zeros((slice_max_prj, 12))
            tmp_proj_data = np.zeros((self.det_size_y, slice_max_prj, self.det_size_x))
            k = 0
            for i in sorted(container):
                if i < total_min_prj:
                    continue
                tmp_vec_geom[k,:] = pre_calc_vec_geom[i,:]
                angle = 2 * 3.1415926*i/self.angles_per_rotation
                z = self.start_scan_position - rec_size_z / 2 - rec_start + (i/self.angles_per_rotation) * self.pitch_dist
                tmp_vec_geom[k, 2] = z
                tmp_vec_geom[k, 5] = z
                tmp_proj_data[:,k,:] = pre_calc_proj_data[:,i,:]
                k += 1
            tmp_proj_data = tmp_proj_data[:,:k,:]
            tmp_vec_geom = tmp_vec_geom[:k,:]
            
            # astra-toolbox routines for recoinstruction using FDK algorithm as a starting point
            # for a SIRT algorith with 500 iteration
            
            proj_geom = astra.create_proj_geom('cone_vec', self.det_size_y, self.det_size_x, tmp_vec_geom)
            proj_id = astra.data3d.create('-proj3d', proj_geom, tmp_proj_data)
            vol_geom = astra.create_vol_geom(self.size_y, self.size_x, rec_size_z)
            rec_id = astra.data3d.create('-vol', vol_geom)

            cfg_precalc =  astra.astra_dict('FDK_CUDA')
            cfg_precalc['ReconstructionDataId'] = rec_id
            cfg_precalc['ProjectionDataId'] = proj_id
            alg_precalc_id = astra.algorithm.create(cfg_precalc)

            astra.algorithm.run(alg_precalc_id)
            
            rec_id_2 = astra.data3d.create('-vol', vol_geom, astra.data3d.get(rec_id)*3.1415926535)
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = rec_id_2
            cfg['ProjectionDataId'] = proj_id
            alg_id = astra.algorithm.create(cfg)
            
            astra.algorithm.run(alg_id, 500)
            
            rec = astra.data3d.get(rec_id_2)
            astra.data2d.delete(rec_id)
            astra.projector.delete(proj_id)
            astra.algorithm.delete(alg_precalc_id)
            astra.algorithm.delete(alg_id)
            
            # intermediate result saving and NN prediction calculation
            self.predict_and_save_rec(rec, len(container), cur_sectors_count, rec_start)  
            
            astra.data2d.delete(rec_id_2)


if __name__ == "__main__":
    data_folder = sys.argv[1]
    NN_path = sys.argv[2]
    patient_full_name = sys.argv[3]
    Patient.data_folder = data_folder
    layer_to_rec = int(sys.argv[4])
    reproection_save_path = sys.argv[5]
    skip_type = sys.argv[6]
    predictions_save_path = sys.argv[7]    
    path_to_log = sys.argv[8]
    stopping_interval = 10
    try:
        os.makedirs(reproection_save_path)
    except OSError as error:
        pass
    experiment = Experiment(patient_full_name, layer_to_rec, reproection_save_path, predictions_save_path, NN_path, skip_type, stopping_interval)

    start = tmr()
    experiment.run()
    end = tmr()
    time = end - start
    print_to_file(f"{patient_full_name} slice {layer_to_rec} exec_time, s: {time}")
    print("0", file = sys.stderr)
    