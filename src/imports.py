## ----- OS ----- ##
import io
import os
import time
import psutil
import random
from natsort import os_sorted, natsorted

## ----- MATH / STATS ----- ##
import cv2
import heapq
import mat73
import scipy.io
import numpy as np  
from copy import deepcopy
from itertools import combinations
from collections import defaultdict
from scipy.optimize import curve_fit
from skimage.draw import line_aa, line
# from sklearn.neighbors import KernelDensity

## ----- TORCH ----- ##
import gc
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset

## ----- FUNCTIONS ----- ##
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/ddp/")
sys.path.insert(0, os.getcwd() + "/ddp/tusnet/")
from functions import *
from plotting import *
from data import *
from metrics import *

## ----- TUSNET ----- ##
from dataset import *
from LSTMConvBlock import *
from LSTMConvCell import *
from PhaseDecoder import *
from PressureDecoder import *

## ----- PLOTTING ----- ##
import matplotlib as mpl
from cycler import cycler
from IPython import display
from tqdm.auto import trange
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

## ----- REPRODUCIBILITY ----- ##
seed = 111
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False