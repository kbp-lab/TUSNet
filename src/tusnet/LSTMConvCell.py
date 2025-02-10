## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

import torch
import torch.nn as nn

## ------------------------------------------------------------
## Explain the block
## ------------------------------------------------------------

## ----- LSTM-LSTM-Conv with Multiple Conv Layers -----
class LSTMConvCell(nn.Module):
    def __init__(self, gpu_id, input_size, hidden_size, num_layers, mode):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        self.gpu_id = gpu_id
        
        self.primary_channels = 8
        self.secondary_channels = 16
    
        if self.mode == 'encode':
            self.lstm_first = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                           batch_first=True)
            self.lstm_second = nn.LSTM(self.input_size, self.hidden_size,
                                     self.num_layers, batch_first=True)
            self.lstm_third = nn.LSTM(self.input_size, self.hidden_size,
                                     self.num_layers, batch_first=True)   
            self.lstm_fourth = nn.LSTM(self.input_size, self.hidden_size,
                                     self.num_layers, batch_first=True)   
            
            ########## uncomment if hidden size = 512 ##########
            self.conv2d_1 = nn.Conv2d(in_channels=4, out_channels=self.primary_channels,
                                  kernel_size=3, stride=1, padding=1)
            self.conv2d_2 = nn.Conv2d(in_channels=self.primary_channels, 
                              out_channels=self.secondary_channels,
                   kernel_size=4, stride=2, padding=1)
            
#             ########## uncomment if hidden size = 1024 ##########                                
#             self.conv2d_1 = nn.Conv2d(in_channels=2, out_channels=self.primary_channels,
#                               kernel_size=(5, 4), stride=(1, 2), padding=(2, 1))
#             self.conv2d_2 = nn.Conv2d(in_channels=self.primary_channels, 
#                                       out_channels=self.secondary_channels,
#                               kernel_size=4, stride=2, padding=1)
                
            self.conv1d = nn.Conv1d(in_channels=self.num_layers, out_channels=self.num_layers,
                              kernel_size=4, stride=2, padding=1)


        elif self.mode == 'decode':
            self.lstm_first = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, 
                             batch_first=True)
            self.lstm_second = nn.LSTM(self.input_size, self.hidden_size,
                                     self.num_layers, batch_first=True)
            self.lstm_third = nn.LSTM(self.input_size, self.hidden_size,
                                     self.num_layers, batch_first=True)   
            self.lstm_fourth = nn.LSTM(self.input_size, self.hidden_size,
                                     self.num_layers, batch_first=True)  
            
            ######### uncomment if hidden size = 512 ##########
            self.conv2d_1 = nn.ConvTranspose2d(in_channels=4, out_channels=self.primary_channels, 
                                               kernel_size=5, stride=1, padding=2)
            self.conv2d_2 = nn.ConvTranspose2d(in_channels=self.primary_channels, 
                                           out_channels=self.secondary_channels, 
                              kernel_size=4, stride=2, padding=1) 

#             ########## uncomment if hidden size = 1024 ##########                
#             self.conv2d_1 = nn.ConvTranspose2d(in_channels=2, out_channels=self.primary_channels, 
#                                                kernel_size=5, stride=1, padding=2)
#             self.conv2d_2 = nn.ConvTranspose2d(in_channels=self.primary_channels, 
#                                            out_channels=self.secondary_channels, 
#                                                kernel_size=(4, 3), stride=(2, 1), padding=1)                                      
                
            self.conv1d = nn.ConvTranspose1d(in_channels=self.num_layers, out_channels=self.num_layers,
                              kernel_size=4, stride=2, padding=1)
        
        self.pooling = nn.Conv2d(in_channels=self.secondary_channels, out_channels=1, 
                                  kernel_size=1, stride=1, padding=0)
        
        self.drop_layer = nn.Dropout2d(p=0.2)
        
    @staticmethod
    def hc_permute(inp):
        '''
        this static method takes the hidden state as well as the cell state
        and change their dimensions with appripriate permutations so that they 
        have the dimensions suitable for convolution
        '''
        inp = inp.reshape(1, *inp.size())
        inp = inp.permute(2, 0, 1, 3)
        
        return inp
    
    def forward(self, input_):
        x = input_[0].squeeze(dim=1) # original
        x_flip90 = torch.rot90(x, k=1, dims=[1, 2])
        x_flip180 = torch.flip(x, dims=[1])
        x_flip270 = torch.flip(x_flip90, dims=[1])
        h0 = input_[1].contiguous()
        c0 = input_[2].contiguous()
            
        # flatten parameters for multi-GPU
        self.lstm_first.flatten_parameters()
        self.lstm_second.flatten_parameters()
        self.lstm_third.flatten_parameters()
        self.lstm_fourth.flatten_parameters()
            
        # apply LSTM to both normal and flipped inputs
        out1, (h11, c11) = self.lstm_first(x, (h0, c0))
        out2, (h12, c12) = self.lstm_second(x_flip90, (h0, c0))   
        out3, (h13, c13) = self.lstm_third(x_flip180, (h0, c0))  
        out4, (h14, c14) = self.lstm_fourth(x_flip270, (h0, c0))   
        
        # add a singleton axis to the lstm output to match the format required by conv2d
        N, H, W = out1.shape[0], out1.shape[1], out1.shape[2]        
        out1 = out1.reshape(N, 1, H, W)
        out2 = out2.reshape(N, 1, H, W)
        out3 = out3.reshape(N, 1, H, W)
        out4 = out4.reshape(N, 1, H, W)
        
        # unflip all outputs
        out2 = torch.rot90(out2, k=-1, dims=[2, 3])
        out3 = torch.flip(out3, dims=[2])
        out4 = torch.rot90(torch.flip(out4, dims=[2]), k=-1, dims=[2, 3])
        
        # concatenate the outputs along the channel dimension. 
        out = torch.cat((out1, out2, out3, out4), dim=1)   # Nx4xHxW
        
        # apply permutations and extensions to hidden and cell states
        h11_prm = LSTMConvCell.hc_permute(h11)   # N x 1 x num_layers x hidden_size
        h12_prm = LSTMConvCell.hc_permute(h12)
        h13_prm = LSTMConvCell.hc_permute(h13)
        h14_prm = LSTMConvCell.hc_permute(h14)
        
        c11_prm = LSTMConvCell.hc_permute(c11)   # N x 1 x num_layers x hidden_size
        c12_prm = LSTMConvCell.hc_permute(c12)
        c13_prm = LSTMConvCell.hc_permute(c13)
        c14_prm = LSTMConvCell.hc_permute(c14)
        
        # concatenate the h and c along the channel dimension 
        h1 = torch.cat((h11_prm, h12_prm, h13_prm, h14_prm), dim=1)
        c1 = torch.cat((c11_prm, c12_prm, c13_prm, c14_prm), dim=1)        
        
        # sum along the channel dimension    
        h1 = torch.sum(h1, axis=1).to(self.gpu_id)   # N x num_layers x hidden_size
        c1 = torch.sum(c1, axis=1).to(self.gpu_id)     
        
        # apply the 1D convolution to h1 and c1
        hc_conv_layer = nn.Sequential(self.conv1d,
                                      nn.ReLU(),
                                      nn.BatchNorm1d(num_features=self.num_layers, device=self.gpu_id))
    
        h1_conv = hc_conv_layer(h1)
        c1_conv = hc_conv_layer(c1)        
        
        # reshape the hidden and cell states into dimensions suitable for subsequent LSTMs
        h1_conv = torch.transpose(h1_conv, 0, 1)   # num_layers x N x hidden_size
        c1_conv = torch.transpose(c1_conv, 0, 1)
        
        # apply the 2D convolution to out3
        out_conv_layer = nn.Sequential(self.conv2d_1,
                                       nn.BatchNorm2d(num_features=self.primary_channels, device=self.gpu_id),
                                       nn.ReLU(),
                                       self.conv2d_2,
                                       nn.BatchNorm2d(num_features=self.secondary_channels, device=self.gpu_id),
                                       nn.ReLU(),  
                                       self.drop_layer,
                                       self.pooling)
        
        out_conv = out_conv_layer(out)   # N x 1 x H x W

        ReLU = nn.ReLU()
        out_conv = ReLU(out_conv)

        return out_conv, h1_conv.contiguous(), c1_conv.contiguous()