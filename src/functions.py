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
from scipy import ndimage
from scipy.ndimage import rotate

## ----- TORCH ----- ##
import torch
import torch.nn as nn 
import torch.optim as optim

## ----- PLOTTING ----- ##
from tqdm.auto import trange
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter


## ------------------------------------------------------------
## Helper functions to convert between numpy arrays and tensors
## ------------------------------------------------------------
device = torch.device("cuda")
to_t = lambda array: torch.tensor(array, device=device)
from_t = lambda tensor: tensor.to('cpu').detach().numpy()


## ------------------------------------------------------------
## Get indices of n largest elements in an array
## ------------------------------------------------------------
def indices_of_n_largest(arr, n):
    return heapq.nlargest(n, range(len(arr)), key=lambda i: arr[i])

## ------------------------------------------------------------
## Get indices of n largest elements in an array
## ------------------------------------------------------------
def indices_of_n_smallest(arr, n):
    return heapq.nsmallest(n, range(len(arr)), key=lambda i: arr[i])

## ------------------------------------------------------------
## Print all tensors stored in GPU memory
## ------------------------------------------------------------
def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))

def dump_tensors(gpu_only=True, print_obj=False):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    if print_obj:
                        print("%s:%s%s %s" % (type(obj).__name__, 
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    if print_obj:
                        print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                   type(obj.data).__name__, 
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "", 
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    return total_size
    

## ------------------------------------------------------------
## Euclidean distance
## ------------------------------------------------------------
def calculate_distance(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


## ------------------------------------------------------------
## Segment largest island in binary mask
## ------------------------------------------------------------
def extract_largest_island(binary_mask):
    # Label connected components in the binary mask
    labeled_mask, num_labels = ndimage.label(binary_mask)

    # Count the number of pixels for each connected component (island)
    island_sizes = np.bincount(labeled_mask.ravel())

    # Set the background (label 0) size to 0
    island_sizes[0] = 0

    # Find the label of the largest island
    largest_island_label = np.argmax(island_sizes)

    # Create a mask with only the largest island
    largest_island_mask = (labeled_mask == largest_island_label).astype(int)

    return largest_island_mask



def unwrap_metrics(file_path):
    
    # load metrics
    data = np.load(file_path)
    keys = data.files
    
    # assert the format is correct (since the keys are hardcoded)
    assert 'focal_pressure_error' in keys[0]
    assert 'focal_volume_iou' in keys[-1]
    
    # Unwrap individual measures
    focal_pressure_error = data[keys[0]]
    focal_pressure_error_raw = np.array(data[keys[1]]) / 1e3

    peak_pressure_error = data[keys[2]]
    peak_pressure_error_comb = data[keys[3]]
    peak_pressure_error_comb_raw = np.array(data[keys[4]]) / 1e3

    focal_location_error = data[keys[5]]
    focal_location_error_x = data[keys[6]]
    focal_location_error_y = data[keys[7]]

    focal_volume_error = data[keys[8]]
    focal_volume_iou = data[keys[9]]
    
    return focal_pressure_error, focal_pressure_error_raw, \
           peak_pressure_error, peak_pressure_error_comb, peak_pressure_error_comb_raw, \
           focal_location_error, focal_location_error_x, focal_location_error_y, \
           focal_volume_error, focal_volume_iou


## ------------------------------------------------------------
## Average distance between the points in two binary masks
## ------------------------------------------------------------
def modified_hausdorff_distance(mask1, mask2, plot = False):
    
    mask1 = extract_largest_island(mask1)
    mask2 = extract_largest_island(mask2)
    
    # Find boundary points using Canny edge detection
    edges1 = cv2.Canny(np.uint8(mask1), 0, 1)
    edges2 = cv2.Canny(np.uint8(mask2), 0, 1)
    
    if plot == True:
         
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(edges1)
        plt.axis("off")
        plt.title("Edges 1")
        
        plt.subplot(1, 2, 2)
        plt.imshow(edges2)
        plt.axis("off")
        plt.title("Edges 2")
        
        plt.tight_layout()
        plt.show()
    
    # Extract coordinates of the boundary points
    boundary_pts1 = np.argwhere(edges1 > 0)
    boundary_pts2 = np.argwhere(edges2 > 0)

    # Compute the distances between all pairs of boundary points
    distances = np.linalg.norm(boundary_pts1[:, None] - boundary_pts2, axis=2)

    # Calculate the average minimum distances for both sets of boundary points
    avg_min_distance1 = np.mean(np.min(distances, axis=1))
    avg_min_distance2 = np.mean(np.min(distances, axis=0))

    # Compute the modified Hausdorff distance
    hausdorff_distance = max(avg_min_distance1, avg_min_distance2)

    return hausdorff_distance
    

## ------------------------------------------------------------
## k-Wave function to convert CT HU to density (kg/m³)
## ------------------------------------------------------------
def hounsfield2density(ct_data):
    # create empty density matrix
    density = np.zeros_like(ct_data)

    # apply conversion in several parts using linear fits to the data
    # Part 1: Less than 930 Hounsfield Units
    density[ct_data < 930] = np.polyval([1.025793065681423, -5.680404011488714], ct_data[ct_data < 930])

    # Part 2: Between 930 and 1098 (soft tissue region)
    density[(ct_data >= 930) & (ct_data <= 1098)] = np.polyval([0.9082709691264, 103.6151457847139], 
                                                               ct_data[(ct_data >= 930) & (ct_data <= 1098)])

    # Part 3: Between 1098 and 1260 (between soft tissue and bone)
    density[(ct_data > 1098) & (ct_data < 1260)] = np.polyval([0.5108369316599, 539.9977189228704], 
                                                              ct_data[(ct_data > 1098) & (ct_data < 1260)])

    # Part 4: Greater than 1260 (bone region)
    density[ct_data >= 1260] = np.polyval([0.6625370912451, 348.8555178455294], ct_data[ct_data >= 1260])

    # calculate corresponding sound speed values
    sound_speed = 1.33 * density + 167

    return density, sound_speed


## ------------------------------------------------------------
## Calculate skull temperature rise with some assumptions
## ------------------------------------------------------------
def calculate_temp_rise(peak_pressure, skull_HU):
    # Convert HU to density and sound speed
    p, c = hounsfield2density(np.array(skull_HU))

    # Calculate Intensity
    I = np.power(peak_pressure, 2) / (2 * p * c)  # W/m^2

    # Calculate absorption coefficient in Np/m
    alpha = 1.53 * 100 * 0.5 # Np/m

    # Calculate heat source
    Q = 2 * alpha * I # W/m^3

    # Specific heat capacity in J/kg/K
    C = 1260 

    # Calculate temperature rise in K
    DeltaT = Q / (p * C) # K

    return DeltaT

def generate_target_coords(idx):
    target_x = (idx - 1) % 8
    target_y = (idx - 1) // 8
    
    target_x = 32 * (target_x + 7)
    target_y = 32 * (target_y + 5)

    return (target_y, target_x)

def get_in_between_targets(target_number1, target_number2):
    # Check if the target numbers are within the valid range
    if not (1 <= target_number1 <= 56) or not (1 <= target_number2 <= 56):
        raise ValueError("Target numbers must be between 1 and 56")
        
    # Check if the target numbers are adjacent in the same row
    if abs(target_number1 - target_number2) != 1 and abs(target_number1 - target_number2) != 8:
        raise ValueError("Targets must be adjacent in the same row")

    # Get the actual coordinates from the base_targets
    target1 = generate_target_coords(target_number1)
    target2 = generate_target_coords(target_number2)
    
    # Calculate the squares / farthest-away points
    if target1[0] == target2[0]: # x-coordinates are equal
        farthest_x1 = target1[0] + 16
        farthest_y1 = int((target1[1] + target2[1]) / 2) - 32
    elif target1[1] == target2[1]: # y-coordinates are equal
        farthest_x1 = int((target1[0] + target2[0]) / 2)
        farthest_y1 = target1[1] + 16
    
    # Calculate the midpoint and quarter-points, rounding to the nearest integer
    mid_x = int((target1[0] + target2[0]) / 2)
    mid_y = int((target1[1] + target2[1]) / 2)

    quarter_x1 = int((3 * target1[0] + target2[0]) / 4)
    quarter_y1 = int((3 * target1[1] + target2[1]) / 4)

    quarter_x2 = int((target1[0] + 3 * target2[0]) / 4)
    quarter_y2 = int((target1[1] + 3 * target2[1]) / 4)

    return [(farthest_x1, farthest_y1), (mid_x, mid_y), (quarter_x1, quarter_y1), (quarter_x2, quarter_y2)], [0.707, 0.5, 0.25, 0.25]

from itertools import combinations

def generate_target_combinations(target_numbers):
    valid_combinations = []

    for combo in combinations(target_numbers, 2):
        # Check if the targets are in the same column and are adjacent
        if abs(combo[0] - combo[1]) == 8:
            valid_combinations.append(combo)
        # Check if the targets are in the same row and are adjacent
        elif (abs(combo[0] - combo[1]) == 1) and ((max(combo) - 1) // 8 == (min(combo) - 1) // 8):
            valid_combinations.append(combo)

    return valid_combinations


## ------------------------------------------------------------
## Count or display parameters in an input model
## ------------------------------------------------------------
def count_parameters(model, print_params=False):
    if print_params:
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## ------------------------------------------------------------
## Convert num_bytes (int) to human-readable format
## ------------------------------------------------------------
def bytes_to_human_readable(num_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"
    
    
## ------------------------------------------------------------
## Visualize model gradients over training epochs
## ------------------------------------------------------------
def grad_visualize(grad_dict, layer_name, num_epochs, bins):
    epochs = [0, 1, 2]
    labels = [1, int(num_epochs/2), num_epochs]
    for epoch in epochs:
        grad = torch.ravel(grad_dict[layer_name][epoch])
        plt.hist(from_t(grad), bins=bins, edgecolor='k', alpha=0.7,
                 label='$\sum$ ' + 'epoch ' + str(labels[epoch]) + ': '  + 
                 str(np.round(torch.sum(abs(grad)).item(), 5)))
        plt.legend()
    plt.show()    


## ------------------------------------------------------------
## Visualize model weights
## ------------------------------------------------------------
def weight_visualize(weight_dict, layer_name, num_epochs):
    weight1 = weight_dict[layer_name][0]
    weight2 = weight_dict[layer_name][-1] 
    weight3 = weight1 - weight2
    
    fig = plt.figure(constrained_layout=True, figsize=(15, 6))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[:2, 1])
    ax3 = fig.add_subplot(gs[:2, 2])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])    
        
    plt1 = ax1.imshow(from_t(weight1), aspect='auto', cmap='RdBu')
    ax1.set_title('weights, epoc 1')
    fig.colorbar(plt1, ax=ax1)
    plt2 = ax2.imshow(from_t(weight2), aspect='auto', cmap='RdBu')
    ax2.set_title('weights, epoc ' + str(num_epochs))
    fig.colorbar(plt2, ax=ax2)
    plt3 = ax3.imshow(from_t(weight3), aspect='auto', cmap='hot')
    ax3.set_title('weights, difference')
    fig.colorbar(plt3, ax=ax3)

    plt4 = ax4.hist(from_t(torch.ravel(weight1)), bins=100, color='deepskyblue')
    plt5 = ax5.hist(from_t(torch.ravel(weight2)), bins=100, color='deepskyblue')
    plt6 = ax6.hist(from_t(torch.ravel(weight3)), bins=100, color='maroon')

    plt.show()