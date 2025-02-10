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


## ------------------------------------------------------------
## Compute TUSNet metrics
## ------------------------------------------------------------

def tusnet_metrics(gt_outputs, net_outputs, gt_pressures, net_pressures, threshold = 0.50, crop = 150, res = 0.1171875):

    """
    Function to compute all evaluation metrics for TUSNet simulations.

    Input parameters:
    - gt_outputs: ground truth pressure fields
    - net_outputs: TUSNet pressure fields
    - gt_pressures: ground truth absolute peak pressures
    - net_pressures: TUSNet absolute peak pressures
    - threshold: peak pressure threshold (default 50%)
    - crop: where to crop out the skulls (default 180 pixels)
    - res: simulation resolution (default 0.1171875 mm)

    Outputs:
    - Lots of metrics
    """
    
    # normalization factors
    skull_norm = 2734.531494140625
    field_norms_norm = 725208.1250
    phase_norm = 1.8769439570576196e-05
    
    # initialize metric arrays
    focal_pressure_error = []
    focal_pressure_error_raw = []

    peak_pressure_error = []
    peak_pressure_error_comb = []
    peak_pressure_error_comb_raw = []

    focal_location_error = []
    focal_location_error_mhd = []
    focal_location_error_x = []
    focal_location_error_y = []

    focal_volume_error = []
    focal_volume_iou = []
    
    angles = []
    
    for i in range(len(gt_outputs)):

        # clone ground truth and TUSNet outputs
        out = np.copy(net_outputs[i])
        gt = np.copy(gt_outputs[i])

        # crop
        out[0:crop, :] = 0
        gt[0:crop, :] = 0

        
        ## ----- FOCAL LOCATION ----- ##

        # focal location errors
        out_focus = np.copy(out)
        gt_focus = np.copy(gt)

        # ellipse mask for focal volume error
        out_focus, _, center_out, _ = add_fwxm_ellipse(out_focus, x = threshold)
        gt_focus, _, center_gt, angle_deg = add_fwxm_ellipse(gt_focus, x = threshold)

        # get coordinates of focal spots
        focal_spot_x_net = np.where(out == np.max(out))[0][0]
        focal_spot_y_net = np.where(out == np.max(out))[1][0]

        focal_spot_x_gt = np.where(gt == np.max(gt))[0][0]
        focal_spot_y_gt = np.where(gt == np.max(gt))[1][0]

        # axial and lateral location errors
        fle_x_abs = abs(focal_spot_x_net - focal_spot_x_gt)
        fle_y_abs = abs(focal_spot_y_net - focal_spot_y_gt)

        # peak location error using modified hausdorff distance
        focal_location_error.append(np.sqrt(np.square(fle_x_abs) + np.square(fle_y_abs)) * res)
        focal_location_error_mhd.append(modified_hausdorff_distance(out_focus, gt_focus) * res)
        
        ## ----- AXIAL / LATERAL ----- ##
        
        # compute the cartesian distance between the centers of the two ellipses
        cartesian_distance = abs(center_out - center_gt) * res

        # convert angle from degrees to radians
        angle_rad = np.radians(angle_deg)

        # generate rotation matrix
        projection_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])

        # use the angle to project the cartesian distances over the major and minor axes of the ellipse
        projected_distance = projection_matrix @ cartesian_distance
        
        focal_location_error_x.append(np.abs(projected_distance[0]))
        focal_location_error_y.append(np.abs(projected_distance[1]))
        
        ## ----- CARTESIAN AXIAL / LATERAL ----- ##
        
        # focal_location_error_x.append(fle_x_abs * res)
        # focal_location_error_y.append(fle_y_abs * res)

        ## ----- FOCAL PRESSURE ERROR ----- ##

        # focal pressures
        out_fp_i = out[focal_spot_x_gt, focal_spot_y_gt]
        gt_fp_i = gt[focal_spot_x_gt, focal_spot_y_gt]

        # focal pressure normalizations
        out_p_factor_i = net_pressures[i] * field_norms_norm
        gt_p_factor_i = gt_pressures[i] * field_norms_norm

        # focal pressure errors
        fpe_i = np.abs(((out_fp_i * out_p_factor_i) - (gt_fp_i * gt_p_factor_i)) / (gt_fp_i * gt_p_factor_i))
        fpe_raw_i = np.abs((out_fp_i * out_p_factor_i) - (gt_fp_i * gt_p_factor_i))

        # add to array
        focal_pressure_error.append(fpe_i * 100)
        focal_pressure_error_raw.append(fpe_raw_i)


        ## ----- PEAK PRESSURE ERROR ----- ##

        # peak pressure error (normalized, percentage)
        ppe_i = np.abs((np.max(out) - np.max(gt)) / np.max(gt))

        peak_pressure_error.append(ppe_i)

        # peak pressure error (absolute pressure, percentage)
        ppe_comb_i = np.abs((np.max(out * net_pressures[i] * field_norms_norm) \
                            - np.max(gt * gt_pressures[i] * field_norms_norm)) \
                            / np.max(gt * gt_pressures[i] * field_norms_norm))

        peak_pressure_error_comb.append(ppe_comb_i * 100)

        # peak pressure error (absolute pressure, kPa)
        ppe_comb_i_raw = np.abs(np.max(out * net_pressures[i] * field_norms_norm) \
                            - np.max(gt * gt_pressures[i] * field_norms_norm))

        peak_pressure_error_comb_raw.append(ppe_comb_i_raw)


        ## ----- FOCAL VOLUME ----- ##

        # focal volume error (area)
        difference = abs(out_focus - gt_focus)
        difference = np.count_nonzero(difference == 1) 
        percent_error = difference / np.count_nonzero(gt_focus == 1) * 100
        focal_volume_error.append(percent_error)
        
        # fve = (np.sum(out_focus) - np.sum(gt_focus)) / np.sum(gt_focus)
        # focal_volume_error.append(np.abs(fve) * 100)

        # focal volume error (intersection over union)
        union = np.sum(np.maximum(out_focus, gt_focus))
        out_focus[out_focus == 0] = -1 # destructive operation so that (out == gt) ignores the background
        overlap = np.sum(out_focus == gt_focus)
        focal_volume_iou.append(overlap / union * 100)


        ## ----- DISPLAY ----- ##

        if i > 0 and (i + 1) % 10 == 0:
            print(str(i + 1) + " samples processed", end="\r")
            
    return focal_pressure_error, focal_pressure_error_raw, \
           peak_pressure_error, peak_pressure_error_comb, peak_pressure_error_comb_raw, \
           focal_location_error, focal_location_error_mhd, focal_location_error_x, focal_location_error_y, \
           focal_volume_error, focal_volume_iou

    print("\n" + "Metrics Computed!")
    
    
## ------------------------------------------------------------
## Ellipse Functions
## ------------------------------------------------------------
    
def fwxm_ellipse(field, x):
    ''' function to apply an ellipse over the FW at x% max of the field
    Input
    ------
        field: 2D pressure field  (numpy array)
        x: cutoff threshold (decimal points. For example, x=0.5 will return FWHM)
    Output
    ------
        fwhm_x: minor axis of the ellipse (pixels)
        fwhm_y: major axis of the ellipse (pixels)
        center: coordinates of the center of the ellipse
        angle_degrees: angle of the ellipse
    '''

    field = np.copy(field)
    field_trimmed = field[150:, :]   # Threshold the field past the 150th row to skip the skull
    # Find the indices of the maximum value
    max_idx = np.array(np.unravel_index(np.argmax(field_trimmed), field_trimmed.shape))
    max_idx[0] = max_idx[0] + 150     # Readjust the row component of the index
    max_value = field[max_idx[0], max_idx[1]]

    # Crop the field about the target ellipse both to avoid the skull and to reduce computational load
    cropped_field = field[max_idx[0]-80:max_idx[0]+80, max_idx[1]-80:max_idx[1]+80]
    half_max = max_value * x

    # Compute the angle of the ellipse by first computing the coordinates for points satisfying the FWXM cutoff
    coords = np.column_stack(np.where(cropped_field >= half_max))
    # Compute the covariance matrix for the coordinates and then obtain the two eigenvectors
    cov_matrix = np.cov(coords.T)
    _, eigenvectors = np.linalg.eig(cov_matrix)
    # Calculate the angle between the horizontal (y=0) and the two eigenvectors of the ellipse
    angle = np.arctan2(eigenvectors[1], eigenvectors[0])
    angle_degrees = np.degrees(angle)[0]
    # Rotate the field opposite of the eigenvalue angle, so the ellipse is aligned with the y axis (x=0)
    rotated_field = rotate(cropped_field, angle=-angle_degrees)
    # Compute the center coordiantes as well as the lengths of the minor and major axes of the ellipse
    center = np.array(np.unravel_index(np.argmax(rotated_field), rotated_field.shape))
    fwhm_x = np.sum(rotated_field[center[0], :] >= half_max)  # FWHM along x-axis
    fwhm_y = np.sum(rotated_field[:, center[1]] >= half_max)  # FWHM along y-axis
    # Update the center so it matches the origianl field
    center = max_idx

    return fwhm_x, fwhm_y, center, angle_degrees


def add_fwxm_ellipse(field, x):
    ''' function to create a mask based on the fitted ellipse
    Input
    ------
        field: 2D pressure field  (numpy array)
        x: cutoff threshold (decimal points. For example, x=0.5 will return FWHM)
    Output
    ------
        mask: binary mask (1s where the ellipse is, and 0s everywhere else)
        field: ellipse superimposed over the original field
    '''
    field = np.copy(field)
    fwhm_x, fwhm_y, center, angle_degrees = fwxm_ellipse(field, x)
    
    # Create a blank array with the same dimensions as the field
    mask = np.zeros((field.shape[0], field.shape[1]), dtype=np.uint8)

    # Draw the filled ellipse with the FWHM as the axes lengths
    cv2.ellipse(mask, (center[1], center[0]), (int(fwhm_x / 2), int(fwhm_y / 2)), -angle_degrees, 0, 360, 1, -1)

    # Update the original field with the ellipse filled with ones
    field[mask == 1] = 1     # field now contains the mask
    mask[mask == 1] = 1      # Binary maxk for downstream analysis       

    return mask, field, center, angle_degrees
    
    
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