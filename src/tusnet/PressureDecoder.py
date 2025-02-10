## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

import torch
import torch.nn as nn
from LSTMConvCell import *

## ------------------------------------------------------------
## Explain the block
## ------------------------------------------------------------

class PressureDecoder(nn.Module):
    def __init__(self, gpu_id):
        super(PressureDecoder, self).__init__()
        
        self.gpu_id = gpu_id
        
        # --------------------------FC-LAYER-----------------------------
        
        self.c1 = nn.Conv2d(1, 128, 4, 2, 1)
        self.c2 = nn.Conv2d(128, 256, 4, 2, 1)
        
        self.p1 = nn.Linear(4096, 2048)        
        self.p2 = nn.Linear(2048, 1024)
        self.p3 = nn.Linear(1024, 512)
        self.p4 = nn.Linear(512, 256)
        self.p5 = nn.Linear(256, 128)
        self.p6 = nn.Linear(128, 64)
        self.p7 = nn.Linear(64, 1)
        
        # self.dropout = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # first convert the input to float and reshape it
        x = x.float()
        if len(x.shape) == 2:
            x = torch.reshape(x, (1, 1, x.shape[0], x.shape[1]))
        
        conv_layer = nn.Sequential(self.c1,
                                   nn.BatchNorm2d(num_features=128, device=self.gpu_id),
                                   nn.ReLU(),
                                   self.c2,
                                   nn.BatchNorm2d(num_features=256, device=self.gpu_id),
                                   nn.ReLU())
        
        pres = conv_layer(x)
        pres = pres.reshape(pres.shape[0], -1)
        
        fc_layer = nn.Sequential(self.p1,
                                 self.p2,
                                 self.p3,
                                 self.p4,
                                 self.p5,
                                 self.p6,
                                 self.p7)
        
        pres = fc_layer(pres)

        return pres.squeeze()

# # Phase Decoder w/ LSTMConv
# class PressureDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PressureDecoder, self).__init__()

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
#         self.fc4 = nn.Linear(in_features=128, out_features=1)
        
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
    
    

# # PressureDecoder Embed
# class PressureDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PressureDecoder, self).__init__()
        
#         self.gpu_id = gpu_id
    
#         # primary convolutions on outputs of blocks 1, 2, 3, and 4 (reduced embedding)
#         self.blk1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk4 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
#                      stride=1, padding=1)   
#         self.blk5 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
#                      stride=1, padding=1)        

#         # secondary convolutions applied to concatenated tensors:
#         self.blk2_second = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=4,
#                      stride=2, padding=1)        
#         self.blk3_second = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4,
#                      stride=2, padding=1)      
#         self.blk4_second = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4,
#                      stride=2, padding=1)                

#         # tertiary convolutions applied to the final output
#         self.c1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4,
#                      stride=2, padding=0)
#         self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
#                      stride=1, padding=0)
#         self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
#                      stride=1, padding=0)  
#         self.c4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
#                      stride=1, padding=0)        
        
#         # fully connected layers 
#         self.fc1 = nn.Linear(in_features=256, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=1)        

#     def forward(self, encoded):

#         # unwrap encodings
#         x1 = encoded[4].float()
#         x2 = encoded[3].float()        
#         x3 = encoded[2].float()        
#         x4 = encoded[1].float()        
#         x5 = encoded[0].float()        
        
#         # first block of convolutions
#         primary_conv_blk1 = nn.Sequential(self.blk1, 
#                                          nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk2 = nn.Sequential(self.blk2,
#                                          nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk3 = nn.Sequential(self.blk3, 
#                                          nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk4 = nn.Sequential(self.blk4, 
#                                          nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk5 = nn.Sequential(self.blk5,
#                                          nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         # second block of convolutions
#         secondary_conv_blk2 = nn.Sequential(self.blk2_second,
#                                            nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))
        
#         secondary_conv_blk3 = nn.Sequential(self.blk3_second,
#                                            nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))
        
#         secondary_conv_blk4 = nn.Sequential(self.blk4_second,
#                                            nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))
        
#         # tertiary block of convolutions
#         tertiary_conv_layers = nn.Sequential(self.c1,
#                                              nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(0.2),
                                             
#                                              self.c2,
#                                              nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(0.2),
                                             
#                                              self.c3,
#                                              nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(0.2),
                                             
#                                              self.c4,
#                                              nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(0.2))
        
#         fully_connected_layers = nn.Sequential(self.fc1, self.fc2)
        
        
#         x1_out = primary_conv_blk1(x1)                            # (N, 64, 128, 128)
#         x2_out = primary_conv_blk2(x2)                            # (N, 64, 128, 128)
#         x3_out = primary_conv_blk3(x3)                            # (N, 32, 64, 64)
#         x4_out = primary_conv_blk4(x4)                            # (N, 16, 32, 32)
#         x5_out = primary_conv_blk5(x5)                            # (N, 8, 16, 16)

#         x12_cat = torch.concat((x1_out, x2_out), dim=1)           # (N, 128, 128, 128)
#         x12_catconv = secondary_conv_blk2(x12_cat)                # (N, 32, 64, 64)

#         x123_cat = torch.concat((x12_catconv, x3_out), dim=1)     # (N, 64, 64, 64)
#         x123_catconv = secondary_conv_blk3(x123_cat)              # (N, 16, 32, 32)

#         x1234_cat = torch.concat((x123_catconv, x4_out), dim=1)   # (N, 32, 32, 32)
#         x1234_catconv = secondary_conv_blk4(x1234_cat)            # (N, 8, 16, 16)
        
#         x12345_cat = torch.concat((x1234_catconv, x5_out), dim=1) # (N, 16, 16, 16)
        
#         conv_out = tertiary_conv_layers(x12345_cat)               # (N, 256, 1, 1)
        
#         fc_out = fully_connected_layers(conv_out.squeeze())       # (N, 80)
        
#         phase = fc_out.squeeze()
        
#         return phase