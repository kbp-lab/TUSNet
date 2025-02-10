## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

import torch
import torch.nn as nn
from LSTMConvCell import *

## ------------------------------------------------------------
## Explain the block
## ------------------------------------------------------------

class LSTMConvBlock(nn.Module):
    def __init__(self, gpu_id, num_cells = 10, input_size = 512, hidden_size = 512, num_layers = 4):
        super().__init__()
        '''
        This class builds the network by organizing the LSTM-Conv Cells into encoder
        and decoder blocks. 

        Attributes:
            ngpu: unumber of GPUs to be used
            num_cells: total number of cells for the network. Must be an even integer,
                       as half of the cells will be encoers and the other half decoders.
            input_size: input dimension, in this case (N, 1, 512, 512)
            hidden_size: size of the hidden dimension, in this case (1, N, 512)
            num_layers: number of LSTM layers used
        '''
        self.gpu_id = gpu_id
        
        # store encode and decode cells for later retrieval in the forward method 
        # where we add the skip connections between the encode and decode arms
        self.encode_cells = []
        self.decode_cells = []

        for cell_idx in range(int(num_cells / 2)):
            self.encode_cells.append(
            LSTMConvCell(self.gpu_id,
                         input_size=int(input_size / 2 ** cell_idx), 
                         hidden_size=int(hidden_size / 2 ** cell_idx), 
                         num_layers=num_layers, mode='encode')
            )
        
        for cell_idx in reversed(range(1, int(num_cells / 2) + 1)):
            self.decode_cells.append(
            LSTMConvCell(self.gpu_id,
                         input_size=int(input_size / 2 ** cell_idx), 
                         hidden_size=int(hidden_size / 2 ** cell_idx), 
                         num_layers=num_layers, mode='decode')            
            )
    
        self.encode_block = nn.Sequential(*self.encode_cells)
        self.decode_block = nn.Sequential(*self.decode_cells)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_):
        
        # multiple outputs are always wrapped into a tuple in PyTorch. 
        # We need to unpack the output into x, h0, and c0
        x = input_[0].squeeze()
        if len(x.shape) == 2:
            x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
        h0 = input_[1].contiguous()
        c0 = input_[2].contiguous()
        h0 = torch.transpose(h0, 0, 1)
        c0 = torch.transpose(c0, 0, 1)
        encode_outputs = []
        
        # apply each layer of the encode block one at a time. 
        # We are doing it this way so that we can access individual outputs 
        # for the skip connections
        for cell in self.encode_block:
            # get each output
            (x, h0, c0) = cell((x, h0, c0))
            # save the outputs into encode_outputs
            encode_outputs.append(x)
        
        # reverse the order of the encode outputs to build
        # skip connections with the decode block    
        encode_outputs = list(reversed(encode_outputs))
        for j, cell in enumerate(self.decode_block):
            (x, h0, c0) = cell((x, h0, c0))
            # include skip connections, but do not modify the final output (x)
            if j < len(encode_outputs) - 1:
                x = x.clone() + encode_outputs[j + 1]

        # embedding = torch.clone(encode_outputs[0])
        # del encode_outputs, h0, c0
        del h0, c0
        
        return x, encode_outputs