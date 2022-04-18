#!/usr/bin/env python
# coding: utf-8

import os
import random
from time import time
import subprocess
from timeit import default_timer as tmr

def print_to_file(*args):
    s = ' '.join([str(x) for x in args])
    print(s)
    with open(path_to_log, 'a') as f:
        f.write(s + "\n")

# Paths
# path to the list of slices to process
patients_to_process_path = "your_root_folder/patients_to_process.csv"
# path to dataset
data_folder = "your_root_folder/data/COVID-CTset/"
# path to pretrained NN
NN_path = "your_root_folder/data/trained_networks/FPN-fold1.hdf5"

# count of stops in a single execution
batch_size = 50

# load patients names to process 
patient_names = []
with open(patients_to_process_path, "r") as f:
    for name in f:
        patient_names.append(name[:-1])
        
# initialization for rules parameters
nospike_cutoff_fixed = [30, 35, 40, 45]
threshold_fixed = [0, 0, 0, 0]
nospike_cutoff_monitored = [30, 30, 30, 30]
threshold_monitored = [10, 15, 20, 30]
spike_threshold = 0.8
ns_cutoff = nospike_cutoff_fixed + nospike_cutoff_monitored
th = threshold_fixed + threshold_monitored

# for each set of parameters and each patient
# for each batch consisting of a {batch_size} of stops in a current patient
# run full experiment in subprocess mode
# the results are saved to {pred_name} after each batch is finished
# each experiment lasts several hours, depending on the hardware setup
# all intermediate reconstructions are saved at {reproection_save_path}/{slice_idx}/{sector_idx}.tif

full_time = 0

for nospike_cutoff, threshold in zip(ns_cutoff, th):
    # path to save intermediate reconstructions and the resulting predictions
    reproection_save_path = f"your_root_folder/exp_results/full_experiment_L{nospike_cutoff}t{threshold}/"
    # path to save log
    path_to_log = reproection_save_path + f"log/log{int(time())}.txt"
    # create folder structure
    try: 
        os.makedirs(reproection_save_path + "log/")
    except OSError as error:
        pass
    
    for p_name in patient_names:
        start = tmr()
        # create output and initialize tmp files.
        save_name = p_name.replace('/', '_')
        prj_name = f'{reproection_save_path}projections_{save_name}.csv'
        pred_name = f'{reproection_save_path}predictions_{save_name}.csv'
        with open(prj_name, 'w') as f:
            f.write('projections\n')
        with open(f'{pred_name}', 'w') as f:
            f.write('')
        with open(reproection_save_path + "last_stop.csv", 'w') as f:
            f.write('0,0')
        print_to_file(f"Current rule L{nospike_cutoff}t{threshold}")
        # main execution loop
        batch_idx = 0
        while True:
            result = subprocess.run(["python", "full_experiment_workflow.py", data_folder,      
                                     NN_path, p_name, reproection_save_path, str(batch_size),  
                                     pred_name, prj_name, path_to_log, 
                                     str(nospike_cutoff), str(threshold), str(spike_threshold)
                                    ], stderr=subprocess.PIPE)
            if result.stderr:
                if str(result.stderr)[-4] == "1":
                    pass
                elif str(result.stderr)[-4] == "0":
                    break                
                else:
                    is_ok = False
                    print_to_file("---!!!!---\n" + f"{p_name} at {batch_idx} batch failed with error", result.stderr)
                    break
            else:
                print_to_file("---!!!!---\n" + f"{p_name} at {batch_idx} batch failed unexpectidly without an error")
            batch_idx += 1
        end = tmr()
        p_time = (end - start) / 60
        full_time += p_time
        print_to_file("---------")
        print_to_file(f"patient {p_name} is finished with {batch_idx} batches in {p_time} minutes")
        print_to_file(f"full execution time {full_time} minutes")
        print_to_file("---------")
