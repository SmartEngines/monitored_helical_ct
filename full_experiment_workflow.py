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
    
    def __init__(self, full_name, reproection_save_path, predictions_save_path, prj_save_path, NN_path, stopping_interval, nospike_cutoff, spike_threshold, threshold):
        self.full_name = full_name
        self.reproection_save_path = reproection_save_path
        self.prj_save_path = prj_save_path
        self.predictions_save_path = predictions_save_path
        self.NN_path = NN_path
        self.stopping_interval = stopping_interval
        self.patient = Patient(full_name)
        
        self.nospike_cutoff = nospike_cutoff
        self.spike_threshold = spike_threshold
        self.threshold = threshold
        
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
        
        #load projections and current layer from last batch iteration
        self.prj_container = pd.read_csv(self.prj_save_path, index_col = False)
        self.prj_container = list(self.prj_container["projections"])
        self.prj_container.sort()
        with open(self.reproection_save_path + "last_stop.csv", 'r') as f:
            self.last_stop, self.current_layer = (int(x) for x in f.readline().split(','))
        
        # load saved predictions
        self.predictions = {}
        with open(self.predictions_save_path, 'r') as f:
            while True:
                a = f.readline()
                if a:
                    a = a.split(",")
                    name = a[0]
                    idx = int(a[1])
                    tmp = list([float(x) for x in a[2:]])
                else: 
                    break
                d = self.predictions.get(name, {})
                d.update({idx:tmp})
                self.predictions.update({name: d})

    def save_batch_rec(self, rec, proj_count, start_z, save_count):
        '''
        making and saving a prediction for the {save_count} layers
        along with saving corresponding reconstructions
        '''
        size_z = rec.shape[0]
        h = rec.shape[1]
        w = rec.shape[2]

        pc = f'{proj_count:06}'
        path = f'{self.reproection_save_path}{pc}/{self.patient.full_name}/'
        try: 
            os.makedirs(path)
        except OSError as error:
            pass
        
        # load NN model
        k.clear_session() #clear keras backend
        custom_object={'UpsampleLike': keras_retinanet.layers._misc.UpsampleLike}
        net = load_model(self.NN_path, custom_objects=custom_object)
        d = self.predictions.get(self.patient.full_name, {})
        
        for i in range(max(size_z - save_count, 0), size_z):
            # make predictions for a current layer
            j = start_z + i
            tmp = d.get(j, [])
            im = Image.fromarray(rec[i,:,:].astype(np.float32))
            pred_ind = net.predict(np.expand_dims(np.expand_dims(rec[i,:,:],axis=0),axis=3))[0][0]
            tmp.append(pred_ind)
            d.update({j:tmp})
            # save reconstruction
            im.convert("L")
            name = f"IM{(j+1):05}.tif"
            im.save(f"{path}{name}")
        self.predictions.update({self.patient.full_name: d})
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
    
    def monitored_spike_decision(self):
        '''
        Make decision based on the rule described in the article
        stop if no spike detected after {nospike_cutoff} predictions
        or after {threshold + spike_position} predictions
        '''
        history = self.predictions.get(self.patient.full_name, {})
        meas = history.get(self.current_layer, [])
        if len(meas) < self.nospike_cutoff: 
            return False
        spike_position = -1
        for i in range(len(meas)):
            if meas[i]>self.spike_threshold:
                spike_position = i
                break
        if spike_position == -1:
            return True
        return len(meas) >= spike_position + self.threshold
    
    def add_projections(self, count, add_type):
        '''
        Add {count} of projection indexes to a pool
        based on {add_type}
        '''        
        with open(self.prj_save_path, 'a') as f:
            for i in range(0, count):
                if add_type == 'partial':
                    if i%10 != 4 and i%10 != 7:
                        continue
                cur_prj_idx = self.last_stop + i
                if cur_prj_idx > self.total_max_proj_count:
                    break
                f.write(str(cur_prj_idx) + "\n")
                self.prj_container.append(cur_prj_idx)
        self.last_stop += count
    
    def run(self):
        '''main method'''
        if not self.patient.is_ok:
            return
        
        self.patient.load_images()
        self.finished = False
        max_rec_size = 7 # max z size for the reconstruction to limit the volume
                
        # calculating the max limit of projections needed for reconstruction
        self.total_max_proj_count = int((self.size_z + self.det_size_y * self.det_spacing_y) / (self.pitch_dist) * self.angles_per_rotation)
        self.total_max_proj_count += (self.stopping_interval - self.total_max_proj_count % self.stopping_interval)
        max_stops = int(self.total_max_proj_count / stopping_interval)
        
        # precalculating full projection data
        
        # forming vector of geometry for astra geometry 
        pre_calc_vec_geom = np.zeros((self.total_max_proj_count, 12))
        for i in range(self.total_max_proj_count):
            angle = 2 * 3.1415926535 * i / self.angles_per_rotation
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
        # building forward projection
        pre_calc_vol_geom = astra.create_vol_geom(self.size_y, self.size_x, self.size_z)
        pre_calc_proj_geom = astra.create_proj_geom('cone_vec', self.det_size_y, self.det_size_x, pre_calc_vec_geom)
        pre_calc_proj_id, pre_calc_proj_data = astra.create_sino3d_gpu(self.patient.data, pre_calc_proj_geom, pre_calc_vol_geom)
        
        # main loop for monitored reconstruction
        for l in range(batch_size):
            
            # get limits of projections for a current layer
            min_prj, max_proj = self.get_min_max_prj(self.current_layer)
            
            # go to the next layer if the current scan possition dont see current layer
            if (self.last_stop > max_proj):
                print_to_file("go_next_layer", "current_layer:", self.current_layer)
                self.current_layer += 1
                min_prj, max_proj = self.get_min_max_prj(self.current_layer)
                
            # make decision based on current stopping rule
            # and add the rest projection in a 'partial' mode if decision is True
            skip_count = 0
            decision = self.monitored_spike_decision()
            if decision:
                skip_count = (max(max_proj - self.last_stop, 0)//self.stopping_interval)*self.stopping_interval
                self.current_layer += 1
                self.add_projections(skip_count, 'partial')
                print_to_file("skip:", skip_count, "current_layer:", self.current_layer)
            with open(self.reproection_save_path + "last_stop.csv", 'w') as f:
                f.write(f'{self.last_stop},{self.current_layer}')
            
            # finish experiment if current_layer >= size_z
            if self.current_layer >= self.size_z:
                self.finished = True
                break
            
            # add next {stopping_interval} projections to the pool
            self.add_projections(self.stopping_interval, 'full')
            self.last_stop = min(self.last_stop, self.total_max_proj_count)
            
            # determine the last layer we can reconstruct
            rec_stop = int(((self.last_stop / self.angles_per_rotation) * self.pitch_dist) + self.start_scan_position + self.det_size_y *  self.det_spacing_y / 2) + 1
            
            # skip iteration if no layers can be reconstructed
            if rec_stop <= 0:
                with open(self.reproection_save_path + "last_stop.csv", 'w') as f:
                    f.write(f'{self.last_stop},{self.current_layer}')
                continue
                
            # determine reconstruction region
            rec_stop = min(rec_stop, self.size_z)
            rec_start = max(rec_stop - max_rec_size, 0)
            min_proj_count = int(max(rec_start/(self.pitch_dist) * self.angles_per_rotation, 0))
            rec_size_z = rec_stop - rec_start
            with open(self.reproection_save_path + "last_stop.csv", 'w') as f:
                f.write(f'{self.last_stop},{self.current_layer}')
            
            # setup a new geometry consisted with a precalculated one 
            #but containing only a limited set of projcetion data
            tmp_vec_geom = np.zeros((self.last_stop, 12))
            tmp_proj_data = np.zeros((self.det_size_y, self.last_stop, self.det_size_x))
            k = 0
            for i in sorted(self.prj_container):
                if i < min_proj_count:
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
            save_count = rec_stop - self.current_layer
            self.save_batch_rec(rec, len(self.prj_container), rec_start, save_count)
            astra.data2d.delete(rec_id_2)
            
            # finish experiment if a total_max_proj_count is reached 
            if self.last_stop >= self.total_max_proj_count:
                self.finished = True
                break
        
        # save predictions
        with open(f'{self.predictions_save_path}', 'w') as f:
            for p_name in self.predictions.keys():
                d = self.predictions[p_name]
                for slice_num in d.keys():
                    pred = d[slice_num]
                    s = f'{p_name},{slice_num},{",".join((str(x) for x in pred))}'
                    f.write(s + '\n')
                    
        astra.projector.delete(pre_calc_proj_id)
        return self.finished

if __name__ == "__main__":
    #Paths
    data_folder           = sys.argv[1]
    NN_path               = sys.argv[2]
    patient_full_name     = sys.argv[3]
    reproection_save_path = sys.argv[4]
    batch_size            = int(sys.argv[5])
    predictions_save_path = sys.argv[6]  
    prj_save_path         = sys.argv[7]  
    path_to_log           = sys.argv[8]
    nospike_cutoff        = int(sys.argv[9])
    threshold             = int(sys.argv[10])
    spike_threshold       = float(sys.argv[11])
    
    Patient.data_folder = data_folder
    stopping_interval = 10
    try:
        os.makedirs(reproection_save_path)
    except OSError as error:
        pass
    experiment = Experiment(patient_full_name, reproection_save_path, predictions_save_path, prj_save_path, NN_path, stopping_interval, nospike_cutoff, spike_threshold, threshold)
    start = tmr()
    finished = experiment.run()
    end = tmr()
    time = end - start
    if finished: 
        output = "0"
    else:
        output = "1"
    print_to_file(f"{patient_full_name} batch {experiment.last_stop} exec_time, s: {int(time)}")
    print(output, file = sys.stderr)