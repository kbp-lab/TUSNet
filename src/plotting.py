## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

## ----- MATH / STATS ----- ##
import torch
import numpy as np

## ----- PLOTTING ----- ##
import matplotlib as mpl
from cycler import cycler
from IPython import display
from tqdm.auto import trange
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

## ------------------------------------------------------------
## Custom Style-Sheet for Matplotlib
## ------------------------------------------------------------

# This implements a slightly modified FiveThirtyEight style-sheet
mpl.rcParams.update({
    "axes.axisbelow": True,
    "axes.edgecolor": "#000000",
    "axes.facecolor": "#ffffff",
    "axes.grid": False,
    "axes.labelsize": "large",
    "axes.linewidth": 1.0,
    "axes.prop_cycle": cycler('color', ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']),
    "axes.titlesize": "large",
    "figure.facecolor": "#ffffff",
    "figure.subplot.bottom": 0.07,
    "figure.subplot.left": 0.08,
    "figure.subplot.right": 0.95,
    "font.size": 16.0,
    "grid.color": "#cbcbcb",
    "grid.linestyle": "-",
    "grid.linewidth": 1.0,
    "legend.fancybox": True,
    "lines.linewidth": 4.0,
    "lines.solid_capstyle": "butt",
    "patch.edgecolor": "#ffffff",
    "patch.linewidth": 0.5,
    "savefig.edgecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "svg.fonttype": "path",
    "xtick.major.size": 0.0,
    "xtick.minor.size": 0.0,
    "ytick.major.size": 0.0,
    "ytick.minor.size": 0.0,
})


## ------------------------------------------------------------
## Helper functions to convert between numpy arrays and tensors
## ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
to_t = lambda array: torch.tensor(array, device=device)
from_t = lambda tensor: tensor.to('cpu').detach().numpy()


## ------------------------------------------------------------
## Plot NxN grids of sets of images (skulls, k-Wave, TUSNet)
## ------------------------------------------------------------
def visualize_images(imgs, titles = ['', '', ''], cmaps = ['gray', 'turbo', 'turbo'], n = 3, peak = [False, False, False], cbar = True, spacing = [0.1, 0.5, 0.5], vmax = -1):
    
    """
    Inputs:
        - imgs: A list of images to be plotted.
        - titles (optional): A list of titles for each set of images. 
        - cmaps (optional): A list of colormaps for each set of images. 
        - n (optional): The number of rows and columns in the grid. It can be an integer (NxN grid) or a list of two integers (N1xN2 grid)
        - peak (optional): A list of Boolean values indicating whether to mark the peak point on each image.
        - cbar (optional): A Boolean value indicating whether to show a colorbar on each image. Default is True.
        - spacing (optional): A list of values specifying the horizontal and vertical spacing between subplots. Default is [0.1, 0.5, 0.5].
    
    Output:
        - fig: The generated matplotlib Figure object containing the plotted images.
    """
    
    if n == [2, 1]:
        fig = plt.figure(figsize=(len(imgs) * 4, 7.25))
    else:
        fig = plt.figure(figsize=(len(imgs) * 8, 8))
        
    outer = gridspec.GridSpec(1, len(imgs), wspace = spacing[0])
    
    for i in range(len(imgs)):
        
        if type(n) == list:
            inner = gridspec.GridSpecFromSubplotSpec(n[0], n[1], subplot_spec=outer[i], wspace=spacing[1], hspace = spacing[2])
            idx = n[0] * n[1]
        else:
            inner = gridspec.GridSpecFromSubplotSpec(n, n, subplot_spec=outer[i], wspace=spacing[1], hspace = spacing[2])
            idx = n ** 2
            
        for j in range(idx):
            
            ax = plt.Subplot(fig, inner[j])

            if j == 1:
                ax.set_title(titles[i])
            
            if type(imgs[i]) is torch.Tensor:
                img = from_t(imgs[i])[j].squeeze()
            else:
                img = imgs[i][j].squeeze()
 
            if vmax > 0 and i > 0:
                im = ax.imshow(img, cmap=cmaps[i], vmin = 0, vmax = vmax)
            else:
                im = ax.imshow(img, cmap=cmaps[i])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            if cbar:
                fig.colorbar(im, fraction=0.046, pad=0.04) 
            if peak[i]:
                ind = np.unravel_index(np.argmax(img[180:, :], axis=None), img.shape)
                ax.plot(ind[1], ind[0] + 180, 'o', color='white', mfc='none', linewidth=1.5)
                ax.text(ind[1] + 25, ind[0] + 180 + 12, str(np.round(np.max(img[180:, :]), 2)) + " MPa", color='white', fontsize = 12)
            
    return fig


## ------------------------------------------------------------
## Stack multiple figures vertically and save them to a file
## ------------------------------------------------------------
def combine_vertically(figures, output_file):
    # Convert each figure to a PIL image object
    images = []
    for fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        images.append(crop_image(Image.open(buf).convert('RGB'), tolerance = 0, padding = 20))

    # Find the maximum width and the total height of the combined image
    max_width = max([img.width for img in images])
    total_height = sum([img.height for img in images])

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new('RGB', (max_width, total_height))

    # Paste each image vertically into the combined image
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the combined image to a file
    combined_image.save(output_file)
    
def crop_image(image, tolerance=0, padding=0):
    
    # Apply the threshold to the image
    thresholded_image = image.point(lambda x: 0 if x < 255 - tolerance else 255)

    # Find the bounding box of the content
    bbox = ImageOps.invert(thresholded_image).getbbox()

    # Add the desired padding to the bounding box
    if padding > 0:
        bbox = (
            max(bbox[0] - padding, 0),
            max(bbox[1] - padding, 0),
            min(bbox[2] + padding, image.width),
            min(bbox[3] + padding, image.height),
        )

    # Crop the image using the adjusted bounding box
    cropped_image = image.crop(bbox)

    return cropped_image


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


## ------------------------------------------------------------
## Gaussian Fitting and Plotting
## ------------------------------------------------------------

# Gaussian function
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-((x - mean) ** 2 / (2 * standard_deviation ** 2)))

# Fit standard gaussian
def fit_gaussian_to_data(data, dmin, dmax, bins=10):
    
    # Create histogram and find bin centers
    count, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit Gaussian
    params, params_covariance = curve_fit(gaussian, bin_mids, count, p0=[np.mean(data), np.max(count), np.std(data)])
    
    # Generate fitted data for plotting
    fit_x = np.linspace(dmin, dmax, 10000)
    fit_y = gaussian(fit_x, *params)
    
    return fit_x, fit_y

# Fit gaussian with KernalDensityEstimation
def fit_gaussian_kde(data, dmin, dmax):

    # Find max value of original data
    counts, bin_edges = np.histogram(data, bins=10)
    
    max_count = np.max(counts)
    max_bin_index = np.argmax(counts)
    percentage = (max_count / len(data))

    # Step 1: Perform KDE
    data = data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(data)

    # Step 2: Sample from KDE
    x_vals = np.linspace(dmin, dmax, 10000).reshape(-1, 1)
    log_dens = kde.score_samples(x_vals)
    y_vals = np.exp(log_dens)

    # Normalize KDE
    kde_max = np.max(y_vals)
    y_vals_normalized = y_vals

    # Step 3: Fit Gaussian
    params, _ = curve_fit(gaussian, x_vals.flatten(), y_vals_normalized)
    fit_y = gaussian(x_vals.flatten(), *params)

    # Normalize Gaussian Fit
    fit_max = np.max(fit_y)
    fit_y_normalized = fit_y / fit_max * percentage

    return x_vals.squeeze(), fit_y_normalized.squeeze()

# Plot stacked histograms with prominent edges
def stacked_gaussian(data, labels, units, bins=50, alpha=[0.15, 0.8], colors=['#30a2da', '#fc4f30'], label_loc="right", mlabels=True, fit=[0, 0]):
    """
    Inputs:
    - data: A list of two datasets to be plotted as stacked histograms.
    - labels: A list of two labels for the datasets.
    - units: The units of measurement for the data.
    - bins: The number of bins in the histograms (default: 50).
    - alpha: A list of two values indicating the transparency of the histogram bars (default: [0.15, 0.8]).
    - colors: A list of two colors for the histogram bars (default: ['#30a2da', '#fc4f30']).
    - label_loc: The location of the labels on the plot, either "right" or "left" (default: "right").
    - mlabels: A boolean indicating whether to add min/mean/max labels to the plot (default: True).
    """
    dmin = np.min([np.min(data[0]), np.min(data[1])])
    dmax = np.max([np.max(data[0]), np.max(data[1])])
    
    # Curve fitting
    if fit[0] == 0:
        fit_x, fit_y = fit_gaussian_to_data(data[0], dmin, dmax, bins=bins)
    elif fit[0] == 1:
        fit_x, fit_y = fit_gaussian_kde(data[0], dmin, dmax)
        
    if fit[1] == 0:
        fit_x_kw, fit_y_kw = fit_gaussian_to_data(data[1], dmin, dmax, bins=bins)
    elif fit[1] == 1:
        fit_x_kw, fit_y_kw = fit_gaussian_kde(data[1], dmin, dmax)
        
    # fit_y /= np.max(fit_y_kw)
    # fit_y_kw /= np.max(fit_y_kw)
        
    # For the first dataset
    plt.fill_between(fit_x, fit_y, color=colors[0], alpha=0.2, label=labels[0])
    plt.plot(fit_x, fit_y, linewidth=1, color=colors[0], alpha=1.0)

    # For the second dataset
    plt.fill_between(fit_x_kw, fit_y_kw, color=colors[1], alpha=0.2, label=labels[1])
    plt.plot(fit_x_kw, fit_y_kw, linewidth=1, color=colors[1], alpha=1.0)

    # Add legend with a thicker border
    legend = plt.legend()
    frame = legend.get_frame()
    frame.set_linewidth(2.0)  # Make the border 2 points thick
    
    concat = np.concatenate([np.array(data[0]), np.array(data[1])])
    
    # Set x-lim
    plt.xlim([min(concat), max(concat)])
    plt.ylim([0, plt.ylim()[1]])
    
    # set y-axis to display percentages
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    if mlabels:
    
        # add metric labels
        metric_text(0.55 if label_loc == "right" else 0.15, 0.7, 'TUSNet')
        metric_labels(data[0], units, 0.55 if label_loc == "right" else 0.15)

        # add metric labels
        metric_text(0.85 if label_loc == "right" else 0.45, 0.7, 'TUSNet Phases')
        metric_labels(data[1], units, 0.85 if label_loc == "right" else 0.45)
        
        

        
        
        
        
def draw_text_box(ax, bbox, color='black', linewidth=2, padding=(0.01, 0.01)):
    """
    Draws a text box with padding.

    - ax: The axes on which to draw.
    - bbox: The bounding box of the text in display coordinates.
    - color: The color of the box.
    - linewidth: The line width of the box.
    - padding: A tuple (horizontal_padding, vertical_padding) in axes coordinates.
    """
    # Transform the bbox to axes coordinates
    bbox_axes = Bbox.transformed(bbox, ax.transAxes.inverted())
    
    # Apply padding
    bbox_padded = Bbox.from_extents(bbox_axes.xmin - padding[0],
                                    bbox_axes.ymin - padding[1],
                                    bbox_axes.xmax + padding[0],
                                    bbox_axes.ymax + padding[1])

    # Create and add the rectangle patch
    rect = Rectangle((bbox_padded.xmin, bbox_padded.ymin), bbox_padded.width, bbox_padded.height,
                     fill=False, edgecolor=color, linewidth=linewidth, transform=ax.transAxes)
    ax.add_patch(rect)

def metric_text(x, y, text, color='black', padding=(0.01, 0.02), box = True):
    """
    Places text on the plot with a bordering box.

    - x, y: The x and y coordinates for the text (in axes coordinates).
    - text: The text to display.
    - color: The color of the bordering box.
    - padding: A tuple (horizontal_padding, vertical_padding) specifying the padding around the text.
    """
    ax = plt.gca()
    # Draw the text and get its bounding box
    text_obj = plt.text(x, y, text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, linespacing = 1.5)
    plt.draw()  # This is necessary to force the renderer to calculate text size
    bbox = text_obj.get_window_extent()
    
    if box:
    
        # Draw the bounding box with padding
        draw_text_box(ax, bbox, color=color, padding=padding)

        

# Add min/mean/max labels to plot
def metric_labels(data, units, label_x, color = '#30a2da'):
    """
    Inputs:
    - data: The data for which min/mean/max labels are added to the plot.
    - units: The units of measurement for the data.
    - label_x: The x-coordinate for placing the labels on the plot.
    """
    
    mean = str(np.round(np.mean(data), 2))
    std = str(np.round(np.std(data), 2))
    
    metric_text(label_x, 0.57, "Average Error\n" + mean + " $\pm$ " + std + units, color = color)
    
# Plot a custom histogram with the frequency in percent and statistical labels
def metric_hist(data, bins=50, alpha=0.85, edgecolor='k', linewidth=1, labels=True, label_loc="right", units="", **kwargs):
    """
    Inputs:
    - data: The data to be plotted as a histogram.
    - bins: The number of bins in the histogram (default: 50).
    - alpha: The transparency of the histogram bars (default: 0.85).
    - edgecolor: The color of the edges of the histogram bars (default: 'k').
    - linewidth: The width of the edges of the histogram bars (default: 1).
    - labels: A boolean indicating whether to add min/mean/max labels to the plot (default: True).
    - label_loc: The location of the labels on the plot, either "right" or "left" (default: "right").
    - units: The units of measurement for the data (default: "").
    - **kwargs: Additional keyword arguments to be passed to the plt.hist() function.
    """
    
    # plot histogram with default options
    plt.hist(data, bins=bins, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, weights = np.ones(len(data)) / len(data), **kwargs)
    
    # add labels for min/mean/max if necessary
    if labels: metric_labels(data, units, 0.85 if label_loc == "right" else 0.15)
    
    # set y-axis to display percentages
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
# Plot stacked histograms with prominent edges
def stacked_metric_hist(data, labels, units, bins=50, alpha=[0.15, 0.8], colors=['#30a2da', '#fc4f30'], label_loc="right", mlabels=True):
    """
    Inputs:
    - data: A list of two datasets to be plotted as stacked histograms.
    - labels: A list of two labels for the datasets.
    - units: The units of measurement for the data.
    - bins: The number of bins in the histograms (default: 50).
    - alpha: A list of two values indicating the transparency of the histogram bars (default: [0.15, 0.8]).
    - colors: A list of two colors for the histogram bars (default: ['#30a2da', '#fc4f30']).
    - label_loc: The location of the labels on the plot, either "right" or "left" (default: "right").
    - mlabels: A boolean indicating whether to add min/mean/max labels to the plot (default: True).
    """
    
    # determine common bin edges
    bin_edges = np.histogram_bin_edges(np.concatenate([data[0], data[1]]), bins = bins)
    
    # plot first set of data
    metric_hist(data[0], bins=bin_edges, color=colors[0], edgecolor=colors[0], linewidth=3, alpha=alpha[0], labels = False, histtype='stepfilled')
    metric_hist(data[0], bins=bin_edges, color=colors[0], edgecolor=colors[0], linewidth=3, alpha=alpha[1], labels = False, histtype='step', label = labels[0])
    
    # plot second set of data
    metric_hist(data[1], bins=bin_edges, color=colors[1], edgecolor=colors[1], linewidth=3, alpha=alpha[0], labels = False, histtype='stepfilled')
    metric_hist(data[1], bins=bin_edges, color=colors[1], edgecolor=colors[1], linewidth=3, alpha=alpha[1], labels = False, histtype='step', label = labels[1])
    
    if mlabels:
    
        mean, mean_kw = str(np.round(np.mean(data[0]), 2)), str(np.round(np.mean(data[1]), 2))
        std, std_kw = str(np.round(np.std(data[0]), 2)), str(np.round(np.std(data[1]), 2))
        
        label_x = 0.55 if label_loc == "right" else 0.15
    
        metric_text(label_x, 0.6, "Average Error\n" + mean + " $\pm$ " + std + units, color = colors[0])
        metric_text(label_x + 0.275, 0.6, "Average Error\n" + mean_kw + " $\pm$ " + std_kw + units, color = colors[1])
    
    concat = np.concatenate([np.array(data[0]), np.array(data[1])])
    
    # Set x-lim
    plt.xlim([min(concat), max(concat)])    
    
    plt.legend()
    
def plot_targets(target_sheet, mid_targets = [], mid_distances = []):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.imshow(target_sheet)
    plt.axis('off') 
    
    plt.subplot(1, 2, 2)    
    ax2 = plt.gca() 
    plt.imshow(target_sheet)   
    plt.xlim([120, 392])
    plt.ylim([200, 472])
    plt.axis('off') 

    for target, distance in zip(mid_targets, mid_distances):
        x, y = target
        if distance == 0.707: # square
            ax.add_patch(patches.Rectangle((x-2, y-2), 4, 4, linewidth=1, edgecolor='coral', facecolor='coral'))
            ax2.add_patch(patches.Rectangle((x-2, y-2), 4, 4, linewidth=1, edgecolor='coral', facecolor='coral'))
        elif distance == 0.5: # star
            ax.scatter(x, y, marker='*', color='limegreen', s=50)
            ax2.scatter(x, y, marker='*', color='limegreen', s=50)
        elif distance == 0.25: # triangle
            ax.add_patch(patches.RegularPolygon((x, y), numVertices=3, radius=4, orientation=np.pi, edgecolor='cyan', facecolor='cyan'))
            ax2.add_patch(patches.RegularPolygon((x, y), numVertices=3, radius=4, edgecolor='cyan', facecolor='cyan'))

    plt.show()
    
def box_plot_metric(data, mid_distances, y_label, ax):
    # Organize data into a dictionary
    data_dict = {distance: [] for distance in [0.25, 0.5, 0.707]}
    for metric, distance in zip(data, mid_distances):
        data_dict[distance].append(metric)
    
    # Preparing data for boxplot
    data_to_plot = [data_dict[distance] for distance in data_dict.keys()]
    
    # Create boxplot on the provided Axes object
    ax.boxplot(data_to_plot, patch_artist=True)
    
    # Setting the x-tick labels to the distances
    ax.set_xticks(ticks=range(1, len(data_dict) + 1))
    ax.set_xticklabels(labels=data_dict.keys())
    
    ax.set_xlabel("Distance [mm]")
    ax.set_ylabel(y_label)
        
# Print statistics for a given array
def print_array_stats(arr):
    print("Mean:", np.mean(arr))
    print("Max:", np.max(arr))
    print("Min:", np.min(arr))
    print("Std:", np.std(arr))
    
    