## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

import torch
import torch.nn as nn
from LSTMConvCell import *

## ------------------------------------------------------------
## Explain the block
## ------------------------------------------------------------

# Working / Latest but Rougher PhaseDecoder w/ LSTM Conv
class PhaseDecoder(nn.Module):
    def __init__(self, gpu_id, ch_mult):
        super(PhaseDecoder, self).__init__()

        self.gpu_id = gpu_id
        self.ch_mult = ch_mult  # channel multiplier
        
        self.blk1 = LSTMConvCell(self.gpu_id,
                                 input_size=256, 
                                 hidden_size=256, 
                                 num_layers=self.ch_mult, mode='encode')
        
        self.blk2 = LSTMConvCell(self.gpu_id,
                                 input_size=128, 
                                 hidden_size=128, 
                                 num_layers=self.ch_mult, mode='encode')
        
        self.blk3 = LSTMConvCell(self.gpu_id,
                                 input_size=64, 
                                 hidden_size=64, 
                                 num_layers=self.ch_mult, mode='encode')

        # fully connected layers
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=80)
        
        self.fully_connected_layers = nn.Sequential(self.fc1, self.fc2, self.fc3, self.fc4)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded):

        # unwrap encodings
        x1 = encoded[4].float()

        
        
        h0 = torch.zeros(x1.shape[0], 1, 256).to(self.gpu_id)
        c0 = torch.zeros(x1.shape[0], 1, 256).to(self.gpu_id)
        
        h0 = torch.transpose(h0, 0, 1)
        c0 = torch.transpose(c0, 0, 1)

        x1_out = self.blk1((x1, h0, c0))                          # (N, 1, 128, 128)
        x1_out = self.relu(x1_out[0])
        
        
        
        h0 = torch.zeros(x1_out.shape[0], 1, 128).to(self.gpu_id)
        c0 = torch.zeros(x1_out.shape[0], 1, 128).to(self.gpu_id)
        
        h0 = torch.transpose(h0, 0, 1)
        c0 = torch.transpose(c0, 0, 1)
        
        x2_out = self.blk2((x1_out, h0, c0))                      # (N, 1, 64, 64)
        x2_out = self.relu(x2_out[0])
        
        
        
        h0 = torch.zeros(x2_out.shape[0], 1, 64).to(self.gpu_id)
        c0 = torch.zeros(x2_out.shape[0], 1, 64).to(self.gpu_id)
        
        h0 = torch.transpose(h0, 0, 1)
        c0 = torch.transpose(c0, 0, 1)
        
        x3_out = self.blk3((x2_out, h0, c0))                      # (N, 1, 32, 32)
        x3_out = self.relu(x3_out[0])

        fc_out = self.fully_connected_layers(x3_out.reshape(x3_out.shape[0], -1))       # (N, 80)
        phase = self.sigmoid(fc_out).squeeze()

        return phase
    
    
# Cleaner implementation but the state dictionaries don't match up, so
# gotta either retrain or manually assemble it 

# # Phase Decoder w/ LSTMConv
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()

#         self.gpu_id = gpu_id
#         self.num_layers = 1

#         # Initialize LSTMConvCell blocks in a loop
#         self.lstm_conv_sizes = [256, 128, 64]
#         self.lstm_conv_blocks = nn.ModuleList([
#             LSTMConvCell(self.gpu_id,
#                          input_size=size, 
#                          hidden_size=size, 
#                          num_layers=self.num_layers, mode='encode')
#             for size in self.lstm_conv_sizes
#         ])

#         # fully connected layers
#         self.fc1 = nn.Linear(in_features=1024, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=256)
#         self.fc3 = nn.Linear(in_features=256, out_features=128)
#         self.fc4 = nn.Linear(in_features=128, out_features=80)
        
#         self.fully_connected_layers = nn.Sequential(self.fc1, self.fc2, self.fc3, self.fc4)
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, encoded):
#         x = encoded[4].float()

#         for lstm_conv_block, size in zip(self.lstm_conv_blocks, self.lstm_conv_sizes):
#             h0 = torch.zeros(x.shape[0], 1, size).to(self.gpu_id)
#             c0 = torch.zeros(x.shape[0], 1, size).to(self.gpu_id)
            
#             h0 = torch.transpose(h0, 0, 1)
#             c0 = torch.transpose(c0, 0, 1)
            
#             x, _, _ = lstm_conv_block((x, h0, c0))
#             x = self.relu(x)

#         x = x.reshape(x.shape[0], -1)

#         fc_out = self.fully_connected_layers(x)  # (N, 80)
#         phase = self.sigmoid(fc_out).squeeze()

#         return phase