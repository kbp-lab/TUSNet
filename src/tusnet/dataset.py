## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

import torch
from functions import *
from skimage.draw import line_aa
from torch.utils.data import DataLoader, TensorDataset, Dataset

## ------------------------------------------------------------
## Chunk/Shard-Based Dataset
## ------------------------------------------------------------
    
class ShardCache:
    def __init__(self, max_cached_shards):
        self.max_cached_shards = max_cached_shards
        self.cache = {}

    def get(self, gpu_id, path, shard_path):
        if shard_path not in self.cache:
            if len(self.cache) >= self.max_cached_shards:
                self._unload_oldest_shard()
                
            skulls = torch.load(path + "inputs_traced/" + shard_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
            shard_data = torch.load(path + "pfields/" + shard_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
            
            fields = shard_data[0] #.to(gpu_id)
            phases = shard_data[1] #.to(gpu_id)
            field_norms = shard_data[2] #.to(gpu_id)
            
            # Normalization
            
            skull_norm = 2734.531494140625  # max of real skulls
            skulls[skulls < 25] = 0
            skulls /= skull_norm
            skulls *= 128
            
            fields /= field_norms[:, np.newaxis, np.newaxis]
            
            field_norms_norm = 725208.1250  # max pressure field
            field_norms /= field_norms_norm
            
            phase_norm = 1.8769439570576196e-05
            phases /= phase_norm
            
            self.cache[shard_path] = [skulls, fields, phases, field_norms]
        return self.cache[shard_path]

    def _unload_oldest_shard(self):
        oldest_shard_path = next(iter(self.cache))
        del self.cache[oldest_shard_path]

        
class CustomDataset(Dataset):
    def __init__(self, gpu_id, path, shard_path, shard_cache):
        self.gpu_id = gpu_id
        self.path = path
        self.shard_path = shard_path
        self.shard_cache = shard_cache

    def __getitem__(self, index):
        
        [skulls, fields, phases, field_norms] = self.shard_cache.get(self.gpu_id, self.path, self.shard_path)
        
        # Return the retrieved data sample from the shard, simple!
        return skulls[index], fields[index], phases[index], field_norms[index]

    def __len__(self):
        # shard_data = self.shard_cache.get(self.gpu_id, self.shard_path)
        # print("Len Call")
        return 256 #len(shard_data[0])
    
    
## ------------------------------------------------------------
## Trace lines on skull from transducer edges to target 
## ------------------------------------------------------------
def target_tracer(gpu_id, skull, target_loc, trxd_left_edge, trxd_right_edge):
    """
    Function to trace straight lines from the transducer edges to the target.

    Input parameters:
    - skull: 2D array representing the skull
    - target_loc: Tuple (x, y) representing the target location
    - trxd_left_edge: Tuple (x, y) representing the left edge of the transducer
    - trxd_right_edge: Tuple (x, y) representing the right edge of the transducer

    Output:
    - Updated field with traced lines

    """
    
    to_t = lambda array: torch.tensor(array, device=gpu_id, dtype=torch.float32)
    
    max_HU = 2734.531494140625 # torch.max(field)
    x_coordinates = [trxd_left_edge[0], target_loc[0], trxd_right_edge[0]]
    y_coordinates = [trxd_left_edge[1], target_loc[1], trxd_right_edge[1]]

    rr1, cc1, val1 = line_aa(trxd_left_edge[1], trxd_left_edge[0], target_loc[1], target_loc[0])
    rr2, cc2, val2 = line_aa(trxd_right_edge[1], trxd_right_edge[0], target_loc[1], target_loc[0])

    skull[rr1, cc1] = max_HU / 4 * to_t(val1).float() 
    skull[rr2, cc2] = max_HU / 4 * to_t(val2).float()

    # set the target to the max value of the field:
    skull[target_loc[1], target_loc[0]] = max_HU
    
    return skull

## ------------------------------------------------------------
## Generate transducer mask
## ------------------------------------------------------------
def generate_transducer():
    transducer_mask = np.zeros((512, 512))

    trxd_left_edge = (15, 25)       # [grid points]
    trxd_right_edge = (489, 25)     # [grid points]
    td_loc = [0.3, 3]               # Transducer Location [cm]
    gs = 0.1171875                  # [mm] per grid point
    n_e = 80                        # Number of Elements

    element_width = int(np.floor(512 / (n_e)));
    td_x0 = round(td_loc[0] / (gs / 10)) - 1
    td_y0 = round((td_loc[1] / (gs / 10)) - (n_e * element_width / 2)) - 1

    # Set Transducers
    for j in range(td_y0, td_y0 + n_e * element_width - 1, element_width):
        transducer_mask[td_x0, j] = 1;

    return transducer_mask    

## ------------------------------------------------------------
## TRANSDUCER MASK
## ------------------------------------------------------------
transducer_mask = generate_transducer()

class CustomDatasetTrace(Dataset):
    def __init__(self, gpu_id, inputs, fields, phases, field_norms, target):
        self.inputs = inputs
        self.fields = fields
        self.phases = phases
        self.field_norms = field_norms
        self.target = target
        self.gpu_id = gpu_id

    def __getitem__(self, idx):
        
        to_t = lambda array: torch.tensor(array, device=self.gpu_id, dtype=torch.float32)
        
        skull = to_t(self.inputs[idx // 56]).float()
        field = to_t(self.fields[idx]).float()
        phase = to_t(self.phases[idx]).float()
        field_norm = to_t(self.field_norms[idx]).float()

        segment = int(512 / 16);
        norm = 2734.531494140625
        trxd_left_edge = (15, 25)       # [grid points]
        trxd_right_edge = (489, 25)     # [grid points]

        target_x = (self.target[idx] - 1) % 8
        target_y = (self.target[idx] - 1) // 8
        
        skull[segment * (target_x + 7), segment * (target_y + 5)] = norm;
        
        # Other Variants : Transducer Mask
        skull += to_t(transducer_mask) * norm
        
        # Other Variants : Traced Field
        target_loc = (segment * (target_y + 5), segment * (target_x + 7))
        skull_traced = torch.clone(skull)
        skull_traced = target_tracer(self.gpu_id, skull_traced, target_loc, trxd_left_edge, trxd_right_edge)
        
        # Normalization  
        skull_norm = 2734.531494140625       # max of real skulls
        field_norms_norm = 725208.1250       # max pressure field
        phase_norm = 1.8769439570576196e-05  # max phase delay
        
        # Rescale Skull
        skull_traced[skull_traced < 25] = 0
        skull_traced /= skull_norm
        skull_traced *= 128

        # Rescale Simulations
        field /= field_norm
        field_norm /= field_norms_norm
        phase /= phase_norm
        
        return skull_traced, field, phase, field_norm

    def __len__(self):
        return len(self.fields)