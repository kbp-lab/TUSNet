## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

## ----- OS ----- ##
import os
import gc
import io

## ----- MATH / STATS ----- ##
import cv2
import heapq
import random
import natsort
import scipy.io
import numpy as np

## ------------------------------------------------------------
## Load validation data
## ------------------------------------------------------------

def import_validation_data(current_path):

    # import skulls and simulations
    skulls = np.load(current_path + "/test_set/skulls_test_2k.npy")
    fields = np.load(current_path + "/test_set/pressures_test_2k.npy")
    phases = np.load(current_path + "/test_set/delays_test_2k.npy")

    # convert phases from timesteps to seconds - each skull has a different dt in the simulations
    dts = np.load(current_path + "/test_set/time_steps.npy")
    dts_tiled = np.array([dt for dt in dts for _ in range(56)])

    # calculate pressure field normalizations
    field_norms = np.max(fields, axis=(1, 2))

    # reshape and rescale phases
    phases = phases.squeeze()
    phases *= dts_tiled[:, np.newaxis]

    # define targets
    targets = np.tile(np.arange(1, 57), len(skulls))
    
    return skulls, fields, phases, field_norms, targets

## ------------------------------------------------------------
## Load and unwrap metrics
## ------------------------------------------------------------

def unwrap_metrics(file_path):
    
    # load metrics
    data = np.load(file_path)
    keys = data.files
    
    # assert the format is correct (keys are hardcoded)
    assert 'focal_pressure_error' in keys[0]
    assert 'focal_volume_iou' in keys[-1]
    
    # Unwrap individual measures
    focal_pressure_error = data[keys[0]]
    focal_pressure_error_raw = np.array(data[keys[1]]) / 1e3

    peak_pressure_error = data[keys[2]]
    peak_pressure_error_comb = data[keys[3]]
    peak_pressure_error_comb_raw = np.array(data[keys[4]]) / 1e3

    focal_location_error = data[keys[5]]
    focal_location_error_mhd = data[keys[6]]
    focal_location_error_x = data[keys[7]]
    focal_location_error_y = data[keys[8]]

    focal_volume_error = data[keys[9]]
    focal_volume_iou = data[keys[10]]
    
    return focal_pressure_error, focal_pressure_error_raw, \
           peak_pressure_error, peak_pressure_error_comb, peak_pressure_error_comb_raw, \
           focal_location_error, focal_location_error_mhd, focal_location_error_x, focal_location_error_y, \
           focal_volume_error, focal_volume_iou
