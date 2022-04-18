#!/usr/bin/env python
# coding: utf-8

import os
from time import time
import subprocess

# Paths
# type of dose reduction experiment
skip_type = "full" # full partial
# path to the list of slices to process
slices_to_process_path = "/your_root_folder/target_info/slices_to_process.csv"
# path to save intermediate reconstructions and the resulting predictions
reproection_save_path = f"/your_root_folder/results/slices_experiment_{skip_type}/"
# path to dataset
data_folder = "/your_root_folder/data/COVID-CTset/"
# path to pretrained NN
NN_path = "/your_root_folder/trained_networks/FPN-fold1.hdf5"
# path to save log
path_to_log = reproection_save_path + f"log/log{int(time())}.txt"

def print_to_file(*args):
    s = ' '.join([str(x) for x in args])
    print(s)
    with open(path_to_log, 'a') as f:
        f.write(s + "\n")
        
# create folder structure
try:
    os.makedirs(reproection_save_path + "log/")
except OSError as error:
    pass

# load indexes of slices to process 
patient_names = {}
with open(slices_to_process_path, "r") as f:
    for row in f:
        name, idxs = row.split(':')
        patient_names.update({name:idxs.split(',')})

# for each slice idx run full experiment in subprocess mode
# the results are saved to {pred_name}
# each experiment lasts several hours, depending on the hardware setup
# all intermediate reconstructions are saved at {reproection_save_path}/{slice_idx}/{sector_idx}.tif

for p_name in sorted(patient_names.keys()):
    idxs = patient_names[p_name]
    for slice_idx in sorted(idxs):
        save_name = p_name.replace("/","_")
        pred_name = f'{reproection_save_path}predictions_{save_name}_{slice_idx}.csv'
        is_ok = True
        result = subprocess.run(["python", "slices_experiment_workflow.py", data_folder, NN_path, p_name, slice_idx, reproection_save_path, skip_type, pred_name, path_to_log], stderr=subprocess.PIPE)
        if result.stderr:
            if str(result.stderr)[-4] == "0":
                if len(str(result.stderr))>6:
                    print_to_file("WARNING:\n", result.stderr)
                else:
                    print_to_file(f"{p_name} slice {slice_idx} is finished")
            else:
                print_to_file("---!!!!---\n" + f"{p_name} slice {slice_idx} failed with error", result.stderr)
        else:
            print_to_file("---!!!!---\n" + f"{p_name} slice {slice_idx} failed unexpectidly without an error")

