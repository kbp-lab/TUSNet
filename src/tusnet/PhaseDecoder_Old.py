# Working / Latest but Rougher PhaseDecoder w/ LSTM Conv
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id, ch_mult):
#         super(PhaseDecoder, self).__init__()

#         self.gpu_id = gpu_id
#         self.ch_mult = ch_mult  # channel multiplier
        
#         self.blk1 = LSTMConvCell(self.gpu_id,
#                                  input_size=256, 
#                                  hidden_size=256, 
#                                  num_layers=self.ch_mult, mode='encode')
        
#         self.blk2 = LSTMConvCell(self.gpu_id,
#                                  input_size=128, 
#                                  hidden_size=128, 
#                                  num_layers=self.ch_mult, mode='encode')
        
#         self.blk3 = LSTMConvCell(self.gpu_id,
#                                  input_size=64, 
#                                  hidden_size=64, 
#                                  num_layers=self.ch_mult, mode='encode')

#         # fully connected layers
#         self.fc1 = nn.Linear(in_features=1024, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=256)
#         self.fc3 = nn.Linear(in_features=256, out_features=128)
#         self.fc4 = nn.Linear(in_features=128, out_features=80)
        
#         self.fully_connected_layers = nn.Sequential(self.fc1, self.fc2, self.fc3, self.fc4)
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, encoded):

#         # unwrap encodings
#         x1 = encoded[4].float()

        
        
#         h0 = torch.zeros(x1.shape[0], 1, 256).to(self.gpu_id)
#         c0 = torch.zeros(x1.shape[0], 1, 256).to(self.gpu_id)
        
#         h0 = torch.transpose(h0, 0, 1)
#         c0 = torch.transpose(c0, 0, 1)

#         x1_out = self.blk1((x1, h0, c0))                          # (N, 1, 128, 128)
#         x1_out = self.relu(x1_out[0])
        
        
        
#         h0 = torch.zeros(x1_out.shape[0], 1, 128).to(self.gpu_id)
#         c0 = torch.zeros(x1_out.shape[0], 1, 128).to(self.gpu_id)
        
#         h0 = torch.transpose(h0, 0, 1)
#         c0 = torch.transpose(c0, 0, 1)
        
#         x2_out = self.blk2((x1_out, h0, c0))                      # (N, 1, 64, 64)
#         x2_out = self.relu(x2_out[0])
        
        
        
#         h0 = torch.zeros(x2_out.shape[0], 1, 64).to(self.gpu_id)
#         c0 = torch.zeros(x2_out.shape[0], 1, 64).to(self.gpu_id)
        
#         h0 = torch.transpose(h0, 0, 1)
#         c0 = torch.transpose(c0, 0, 1)
        
#         x3_out = self.blk3((x2_out, h0, c0))                      # (N, 1, 32, 32)
#         x3_out = self.relu(x3_out[0])

#         fc_out = self.fully_connected_layers(x3_out.reshape(x3_out.shape[0], -1))       # (N, 80)
#         phase = self.sigmoid(fc_out).squeeze()

#         return phase





# # Phase Embed Dynamic
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id, ch_mult):
#         super(PhaseDecoder, self).__init__()

#         self.gpu_id = gpu_id
#         self.ch_mult = ch_mult  # channel multiplier

#         # primary convolutions on outputs of blocks 1, 2, 3, and 4 (reduced embedding)
#         self.blk1 = nn.Conv2d(in_channels=1, out_channels=16 * self.ch_mult, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk2 = nn.Conv2d(in_channels=1, out_channels=16 * self.ch_mult, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk3 = nn.Conv2d(in_channels=1, out_channels=8 * self.ch_mult, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk4 = nn.Conv2d(in_channels=1, out_channels=4 * self.ch_mult, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk5 = nn.Conv2d(in_channels=1, out_channels=2 * self.ch_mult, kernel_size=3,
#                      stride=1, padding=1)

#         # secondary convolutions applied to concatenated tensors:
#         self.blk2_second = nn.Conv2d(in_channels=16 * self.ch_mult * 2, out_channels=8 * self.ch_mult, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk3_second = nn.Conv2d(in_channels=8 * self.ch_mult * 2, out_channels=4 * self.ch_mult, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk4_second = nn.Conv2d(in_channels=4 * self.ch_mult * 2, out_channels=2 * self.ch_mult, kernel_size=4,
#                      stride=2, padding=1)

#         # tertiary convolutions applied to the final output
#         self.c1 = nn.Conv2d(in_channels=4 * self.ch_mult, out_channels=32, kernel_size=4,
#                      stride=2, padding=0)
#         self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
#                      stride=1, padding=0)
#         self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
#                      stride=1, padding=0)
#         self.c4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
#                      stride=1, padding=0)

#         # fully connected layers
#         # self.fc1 = nn.Linear(in_features=256, out_features=128)
#         # self.fc2 = nn.Linear(in_features=128, out_features=80)
        
#         self.fc1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, groups=16)
#         self.fc2 = nn.Conv1d(in_channels=128, out_channels=80, kernel_size=1, groups=16)

#     def forward(self, encoded):

#         # unwrap encodings
#         x1 = encoded[4].float()
#         x2 = encoded[3].float()
#         x3 = encoded[2].float()
#         x4 = encoded[1].float()
#         x5 = encoded[0].float()

#         # first block of convolutions
#         primary_conv_blk1 = nn.Sequential(self.blk1,
#                                          nn.BatchNorm2d(num_features=16 * self.ch_mult, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk2 = nn.Sequential(self.blk2,
#                                          nn.BatchNorm2d(num_features=16 * self.ch_mult, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk3 = nn.Sequential(self.blk3,
#                                          nn.BatchNorm2d(num_features=8 * self.ch_mult, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk4 = nn.Sequential(self.blk4,
#                                          nn.BatchNorm2d(num_features=4 * self.ch_mult, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk5 = nn.Sequential(self.blk5,
#                                          nn.BatchNorm2d(num_features=2 * self.ch_mult, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         # second block of convolutions
#         secondary_conv_blk2 = nn.Sequential(self.blk2_second,
#                                            nn.BatchNorm2d(num_features=8 * self.ch_mult, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))

#         secondary_conv_blk3 = nn.Sequential(self.blk3_second,
#                                            nn.BatchNorm2d(num_features=4 * self.ch_mult, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))

#         secondary_conv_blk4 = nn.Sequential(self.blk4_second,
#                                            nn.BatchNorm2d(num_features=2 * self.ch_mult, device=self.gpu_id),
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


#         x1_out = primary_conv_blk1(x1)                            # (N, 16, 128, 128)
#         x2_out = primary_conv_blk2(x2)                            # (N, 16, 128, 128)
#         x3_out = primary_conv_blk3(x3)                            # (N, 8, 64, 64)
#         x4_out = primary_conv_blk4(x4)                            # (N, 4, 32, 32)
#         x5_out = primary_conv_blk5(x5)                            # (N, 2, 16, 16)

#         x12_cat = torch.concat((x1_out, x2_out), dim=1)           # (N, 32, 128, 128)
#         x12_catconv = secondary_conv_blk2(x12_cat)                # (N, 8, 64, 64)
#         x123_cat = torch.concat((x12_catconv, x3_out), dim=1)     # (N, 16, 64, 64)
#         x123_catconv = secondary_conv_blk3(x123_cat)              # (N, 4, 32, 32)
#         x1234_cat = torch.concat((x123_catconv, x4_out), dim=1)   # (N, 8, 32, 32)
#         x1234_catconv = secondary_conv_blk4(x1234_cat)            # (N, 2, 16, 16)
#         x12345_cat = torch.concat((x1234_catconv, x5_out), dim=1) # (N, 4, 16, 16)
#         conv_out = tertiary_conv_layers(x12345_cat)               # (N, 256, 1, 1)

#         # # uncomment below to see the state dimensions
#         # print('primary convolutional block dimensions')
#         # for j in [x1_out, x2_out, x3_out, x4_out, x5_out]:
#         #     print(f'size: {j.shape}')
#         # print('secondary concatenation block dimensions')
#         # for j in [x12_cat, x123_cat, x1234_cat, x12345_cat]:
#         #     print(f'size: {j.shape}')
#         # print('secondary convlotional block dimensions')
#         # for j in [x12_catconv, x123_catconv, x1234_catconv, conv_out]:
#         #     print(f'size: {j.shape}')


#         # fc_out = fully_connected_layers(conv_out.squeeze())       # (N, 80)
#         fc_out = fully_connected_layers(conv_out.squeeze(-1))  

#         phase = fc_out.squeeze()

#         return phase

# # Phase Decoder w/ LSTMConv Test 2
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id, ch_mult):
#         super(PhaseDecoder, self).__init__()

#         self.gpu_id = gpu_id
#         self.ch_mult = ch_mult  # channel multiplier

#         # Initialize LSTMConvCell blocks in a loop
#         self.lstm_conv_sizes = [128, 64, 32]
#         self.lstm_conv_blocks = nn.ModuleList([
#             LSTMConvCell(self.gpu_id,
#                          input_size=size, 
#                          hidden_size=size, 
#                          num_layers=self.ch_mult, mode='encode')
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
#         x = encoded[3].float()

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
    

# # Phase Embed Dense
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()
        
#         self.gpu_id = gpu_id
    
#         # primary convolutions on outputs of blocks 1, 2, 3, and 4 (reduced embedding)
#         self.blk1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
#                      stride=1, padding=1)   
#         self.blk5 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
#                      stride=1, padding=1)        

#         # secondary convolutions applied to concatenated tensors:
#         self.blk2_second = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=4,
#                      stride=2, padding=1)        
#         self.blk3_second = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=4,
#                      stride=2, padding=1)      
#         self.blk4_second = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4,
#                      stride=2, padding=1)                

#         # tertiary convolutions applied to the final output
#         self.c1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
#                      stride=2, padding=0)
#         self.c2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
#                      stride=1, padding=0)
#         self.c3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
#                      stride=1, padding=0)  
#         self.c4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
#                      stride=1, padding=0)        
        
#         # fully connected layers 
#         self.fc1 = nn.Linear(in_features=512, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=80)        

#     def forward(self, encoded):

#         # unwrap encodings
#         x1 = encoded[4].float()
#         x2 = encoded[3].float()        
#         x3 = encoded[2].float()        
#         x4 = encoded[1].float()        
#         x5 = encoded[0].float()        
        
#         # first block of convolutions
#         primary_conv_blk1 = nn.Sequential(self.blk1, 
#                                          nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk2 = nn.Sequential(self.blk2,
#                                          nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk3 = nn.Sequential(self.blk3, 
#                                          nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk4 = nn.Sequential(self.blk4, 
#                                          nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         primary_conv_blk5 = nn.Sequential(self.blk5,
#                                          nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))
        
#         # second block of convolutions
#         secondary_conv_blk2 = nn.Sequential(self.blk2_second,
#                                            nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))
        
#         secondary_conv_blk3 = nn.Sequential(self.blk3_second,
#                                            nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))
        
#         secondary_conv_blk4 = nn.Sequential(self.blk4_second,
#                                            nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))
        
#         # tertiary block of convolutions
#         tertiary_conv_layers = nn.Sequential(self.c1,
#                                              nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(0.2),
                                             
#                                              self.c2,
#                                              nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(0.2),
                                             
#                                              self.c3,
#                                              nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(0.2),
                                             
#                                              self.c4,
#                                              nn.BatchNorm2d(num_features=512, device=self.gpu_id),
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
    
    
    

    
# # Phase Embed
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()
        
#         self.gpu_id = gpu_id
        
#         self.dropout_ratio = 0.3
    
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
#         self.fc2 = nn.Linear(in_features=128, out_features=80)        

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
#                                          nn.Dropout(self.dropout_ratio))
        
#         primary_conv_blk2 = nn.Sequential(self.blk2,
#                                          nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(self.dropout_ratio))
        
#         primary_conv_blk3 = nn.Sequential(self.blk3, 
#                                          nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(self.dropout_ratio))
        
#         primary_conv_blk4 = nn.Sequential(self.blk4, 
#                                          nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(self.dropout_ratio))
        
#         primary_conv_blk5 = nn.Sequential(self.blk5,
#                                          nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(self.dropout_ratio))
        
#         # second block of convolutions
#         secondary_conv_blk2 = nn.Sequential(self.blk2_second,
#                                            nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(self.dropout_ratio))
        
#         secondary_conv_blk3 = nn.Sequential(self.blk3_second,
#                                            nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(self.dropout_ratio))
        
#         secondary_conv_blk4 = nn.Sequential(self.blk4_second,
#                                            nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(self.dropout_ratio))
        
#         # tertiary block of convolutions
#         tertiary_conv_layers = nn.Sequential(self.c1,
#                                              nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(self.dropout_ratio),
                                             
#                                              self.c2,
#                                              nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(self.dropout_ratio),
                                             
#                                              self.c3,
#                                              nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(self.dropout_ratio),
                                             
#                                              self.c4,
#                                              nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                              nn.ReLU(),
#                                              nn.Dropout(self.dropout_ratio))
        
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



# # Phase Embed Lean
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()

#         self.gpu_id = gpu_id

#         # primary convolutions on outputs of blocks 1, 2, 3, and 4 (reduced embedding)
#         self.blk1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk3 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk4 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk5 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
#                      stride=1, padding=1)

#         # secondary convolutions applied to concatenated tensors:
#         self.blk2_second = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk3_second = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk4_second = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4,
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
#         self.fc2 = nn.Linear(in_features=128, out_features=80)

#     def forward(self, encoded):

#         # unwrap encodings
#         x1 = encoded[4].float()
#         x2 = encoded[3].float()
#         x3 = encoded[2].float()
#         x4 = encoded[1].float()
#         x5 = encoded[0].float()

#         # first block of convolutions
#         primary_conv_blk1 = nn.Sequential(self.blk1,
#                                          nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk2 = nn.Sequential(self.blk2,
#                                          nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk3 = nn.Sequential(self.blk3,
#                                          nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk4 = nn.Sequential(self.blk4,
#                                          nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk5 = nn.Sequential(self.blk5,
#                                          nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         # second block of convolutions
#         secondary_conv_blk2 = nn.Sequential(self.blk2_second,
#                                            nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))

#         secondary_conv_blk3 = nn.Sequential(self.blk3_second,
#                                            nn.BatchNorm2d(num_features=8, device=self.gpu_id),
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


#         x1_out = primary_conv_blk1(x1)                            # (N, 32, 128, 128)
#         x2_out = primary_conv_blk2(x2)                            # (N, 32, 128, 128)
#         x3_out = primary_conv_blk3(x3)                            # (N, 16, 64, 64)
#         x4_out = primary_conv_blk4(x4)                            # (N, 8, 32, 32)
#         x5_out = primary_conv_blk5(x5)                            # (N, 8, 16, 16)

#         x12_cat = torch.concat((x1_out, x2_out), dim=1)           # (N, 64, 128, 128)
#         x12_catconv = secondary_conv_blk2(x12_cat)                # (N, 16, 64, 64)
#         x123_cat = torch.concat((x12_catconv, x3_out), dim=1)     # (N, 32, 64, 64)
#         x123_catconv = secondary_conv_blk3(x123_cat)              # (N, 8, 32, 32)
#         x1234_cat = torch.concat((x123_catconv, x4_out), dim=1)   # (N, 16, 32, 32)
#         x1234_catconv = secondary_conv_blk4(x1234_cat)            # (N, 8, 16, 16)
#         x12345_cat = torch.concat((x1234_catconv, x5_out), dim=1) # (N, 16, 16, 16)
#         conv_out = tertiary_conv_layers(x12345_cat)               # (N, 256, 1, 1)

#         fc_out = fully_connected_layers(conv_out.squeeze())       # (N, 80)

#         phase = fc_out.squeeze()

#         return phase


# # Phase Embed Lean 2
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()

#         self.gpu_id = gpu_id

#         # primary convolutions on outputs of blocks 1, 2, 3, and 4 (reduced embedding)
#         self.blk1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk3 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk4 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3,
#                      stride=1, padding=1)
#         self.blk5 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
#                      stride=1, padding=1)

#         # secondary convolutions applied to concatenated tensors:
#         self.blk2_second = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk3_second = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=4,
#                      stride=2, padding=1)
#         self.blk4_second = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=4,
#                      stride=2, padding=1)

#         # tertiary convolutions applied to the final output
#         self.c1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4,
#                      stride=2, padding=0)
#         self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
#                      stride=1, padding=0)
#         self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
#                      stride=1, padding=0)
#         self.c4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
#                      stride=1, padding=0)

#         # fully connected layers
#         self.fc1 = nn.Linear(in_features=256, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=80)

#     def forward(self, encoded):

#         # unwrap encodings
#         x1 = encoded[4].float()
#         x2 = encoded[3].float()
#         x3 = encoded[2].float()
#         x4 = encoded[1].float()
#         x5 = encoded[0].float()

#         # first block of convolutions
#         primary_conv_blk1 = nn.Sequential(self.blk1,
#                                          nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk2 = nn.Sequential(self.blk2,
#                                          nn.BatchNorm2d(num_features=16, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk3 = nn.Sequential(self.blk3,
#                                          nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk4 = nn.Sequential(self.blk4,
#                                          nn.BatchNorm2d(num_features=4, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         primary_conv_blk5 = nn.Sequential(self.blk5,
#                                          nn.BatchNorm2d(num_features=2, device=self.gpu_id),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.2))

#         # second block of convolutions
#         secondary_conv_blk2 = nn.Sequential(self.blk2_second,
#                                            nn.BatchNorm2d(num_features=8, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))

#         secondary_conv_blk3 = nn.Sequential(self.blk3_second,
#                                            nn.BatchNorm2d(num_features=4, device=self.gpu_id),
#                                            nn.ReLU(),
#                                            nn.Dropout(0.2))

#         secondary_conv_blk4 = nn.Sequential(self.blk4_second,
#                                            nn.BatchNorm2d(num_features=2, device=self.gpu_id),
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


#         x1_out = primary_conv_blk1(x1)                            # (N, 16, 128, 128)
#         x2_out = primary_conv_blk2(x2)                            # (N, 16, 128, 128)
#         x3_out = primary_conv_blk3(x3)                            # (N, 8, 64, 64)
#         x4_out = primary_conv_blk4(x4)                            # (N, 4, 32, 32)
#         x5_out = primary_conv_blk5(x5)                            # (N, 2, 16, 16)

#         x12_cat = torch.concat((x1_out, x2_out), dim=1)           # (N, 32, 128, 128)
#         x12_catconv = secondary_conv_blk2(x12_cat)                # (N, 8, 64, 64)
#         x123_cat = torch.concat((x12_catconv, x3_out), dim=1)     # (N, 16, 64, 64)
#         x123_catconv = secondary_conv_blk3(x123_cat)              # (N, 4, 32, 32)
#         x1234_cat = torch.concat((x123_catconv, x4_out), dim=1)   # (N, 8, 32, 32)
#         x1234_catconv = secondary_conv_blk4(x1234_cat)            # (N, 2, 16, 16)
#         x12345_cat = torch.concat((x1234_catconv, x5_out), dim=1) # (N, 4, 16, 16)
#         conv_out = tertiary_conv_layers(x12345_cat)               # (N, 256, 1, 1)

#         fc_out = fully_connected_layers(conv_out.squeeze())       # (N, 80)

#         phase = fc_out.squeeze()

#         return phase
    
    
    
    

# class PressureDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PressureDecoder, self).__init__()
        
#         self.gpu_id = gpu_id
        
#         # --------------------------FC-LAYER-----------------------------
        
#         self.c1 = nn.Conv2d(1, 128, 4, 2, 1)
#         self.c2 = nn.Conv2d(128, 256, 4, 2, 1)
        
#         self.p1 = nn.Linear(4096, 2048)        
#         self.p2 = nn.Linear(2048, 1024)
#         self.p3 = nn.Linear(1024, 512)
#         self.p4 = nn.Linear(512, 256)
#         self.p5 = nn.Linear(256, 128)
#         self.p6 = nn.Linear(128, 64)
#         self.p7 = nn.Linear(64, 1)
        
#         # self.dropout = nn.Dropout(0.2)
#         self.ReLU = nn.ReLU()

#     def forward(self, x):
#         # first convert the input to float 
#         x = x[0].squeeze().float()
#         x = x.reshape(x.shape[0], 1, 16, 16)
        
#         conv_layer = nn.Sequential(self.c1,
#                                    nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c2,
#                                    nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                    nn.ReLU())
        
#         pres = conv_layer(x)
#         pres = pres.reshape(pres.shape[0], -1)
        
#         fc_layer = nn.Sequential(self.p1,
#                                  self.p2,
#                                  self.p3,
#                                  self.p4,
#                                  self.p5,
#                                  self.p6,
#                                  self.p7)
        
#         pres = fc_layer(pres)

#         return pres.squeeze()




# # Phase 4-4-256
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()
        
#         self.gpu_id = gpu_id

#         # --------------------------FC-LAYER-----------------------------
        
#         self.c1 = nn.Conv2d(1, 32, 3, 1, 1) # N x 16 x 8 x 8
#         self.c2 = nn.Conv2d(32, 64, 4, 2, 1) # N x 64 x 2 x 2
#         self.c3 = nn.Conv2d(64, 128, 4, 2, 1) # N x 64 x 2 x 2
#         self.c4 = nn.Conv2d(128, 256, 4, 2, 1) # N x 128 x 1 x 1
        
#         self.p1 = nn.Linear(1024, 512)
#         self.p2 = nn.Linear(512, 256)
#         self.p3 = nn.Linear(256, 128)
#         self.p4 = nn.Linear(128, 80)
        
#         self.ReLU = nn.ReLU()
           
#     def forward(self, x):
#         # first convert the input to float 
#         x = x.squeeze().float()
#         x = x.reshape(x.shape[0], 1, 16, 16)
        
#         conv_layer = nn.Sequential(self.c1,
#                                    nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c2,
#                                    nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c3,
#                                    nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c4,
#                                    nn.BatchNorm2d(num_features=256, device=self.gpu_id))
        
#         phase = conv_layer(x)
#         phase = phase.reshape(phase.shape[0], -1)
        
#         fc_layer = nn.Sequential(self.p1,
#                                  self.p2,
#                                  self.p3,
#                                  self.p4)
        
#         phase = fc_layer(phase)

#         return phase
   
# Phase 5-5-512
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()
        
#         self.gpu_id = gpu_id

#         # --------------------------FC-LAYER-----------------------------
        
#         self.c1 = nn.Conv2d(1, 32, 3, 1, 1) # N x 32 x 16 x 16
#         self.c2 = nn.Conv2d(32, 64, 3, 1, 1) # N x 64 x 16 x 16
#         self.c3 = nn.Conv2d(64, 128, 4, 2, 1) # N x 128 x 8 x 8
#         self.c4 = nn.Conv2d(128, 256, 4, 2, 1) # N x 256 x 4 x 4
#         self.c5 = nn.Conv2d(256, 512, 4, 2, 1) # N x 512 x 2 x 2
        
#         self.p1 = nn.Linear(2048, 1024)
#         self.p2 = nn.Linear(1024, 512)
#         self.p3 = nn.Linear(512, 256)
#         self.p4 = nn.Linear(256, 128)
#         self.p5 = nn.Linear(128, 80)
        
#         self.ReLU = nn.ReLU()
           
#     def forward(self, x):
#         # first convert the input to float 
#         x = x.squeeze().float()
#         x = x.reshape(x.shape[0], 1, 16, 16)
        
#         conv_layer = nn.Sequential(self.c1,
#                                    nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c2,
#                                    nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c3,
#                                    nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c4,
#                                    nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c5,
#                                    nn.BatchNorm2d(num_features=512, device=self.gpu_id))
        
#         phase = conv_layer(x)
#         phase = phase.reshape(phase.shape[0], -1)
        
#         fc_layer = nn.Sequential(self.p1,
#                                  self.p2,
#                                  self.p3,
#                                  self.p4,
#                                  self.p5)
        
#         phase = fc_layer(phase)

#         return phase
    
# # Phase 5-5-512-SD
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()
        
#         self.gpu_id = gpu_id

#         # --------------------------FC-LAYER-----------------------------
        
#         self.c1 = nn.Conv2d(1, 32, 3, 1, 1) # N x 32 x 16 x 16
#         self.c2 = nn.Conv2d(32, 64, 3, 1, 1) # N x 64 x 16 x 16
#         self.c3 = nn.Conv2d(64, 128, 4, 2, 1) # N x 128 x 8 x 8
#         self.c4 = nn.Conv2d(128, 256, 4, 2, 1) # N x 256 x 4 x 4
#         self.c5 = nn.Conv2d(256, 512, 4, 2, 1) # N x 512 x 2 x 2
        
#         self.p1 = nn.Linear(2048, 1024)
#         self.p2 = nn.Linear(1024, 512)
#         self.p3 = nn.Linear(512, 256)
#         self.p4 = nn.Linear(256, 128)
#         self.p5 = nn.Linear(128, 80)
        
#         self.Sigmoid = nn.Sigmoid()
#         self.drop_layer = nn.Dropout2d(p=0.2)
           
#     def forward(self, x):
#         # first convert the input to float 
#         x = x.squeeze().float()
#         x = x.reshape(x.shape[0], 1, 16, 16)
        
#         conv_layer = nn.Sequential(self.c1,
#                                    nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c2,
#                                    nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c3,
#                                    nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c4,
#                                    nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c5,
#                                    nn.BatchNorm2d(num_features=512, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.drop_layer)
        
#         phase = conv_layer(x)
#         phase = phase.reshape(phase.shape[0], -1)
        
#         fc_layer = nn.Sequential(self.p1,
#                                  self.p2,
#                                  self.p3,
#                                  self.p4,
#                                  self.p5,
#                                  self.Sigmoid)
        
#         phase = fc_layer(phase)

#         return phase

# # Phase 5-5-512-Conv
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()
        
#         self.gpu_id = gpu_id

#         # --------------------------FC-LAYER-----------------------------
        
#         self.c1 = nn.Conv2d(1, 32, 3, 1, 1) # N x 32 x 16 x 16
#         self.c2 = nn.Conv2d(32, 64, 3, 1, 1) # N x 64 x 16 x 16
#         self.c3 = nn.Conv2d(64, 128, 4, 2, 1) # N x 128 x 8 x 8
#         self.c4 = nn.Conv2d(128, 256, 4, 2, 1) # N x 256 x 4 x 4
#         self.c5 = nn.Conv2d(256, 512, 4, 2, 1) # N x 512 x 2 x 2
        
#         self.p11 = nn.Conv1d(1, 32, 3, 1, 1)
#         self.p12 = nn.Conv1d(32, 64, 3, 1, 1)
#         self.p13 = nn.Conv1d(64, 1, 4, 2, 1)
        
#         # self.p1 = nn.Linear(2048, 1024)
#         self.p2 = nn.Linear(1024, 512)
#         self.p3 = nn.Linear(512, 256)
#         self.p4 = nn.Linear(256, 128)
#         self.p5 = nn.Linear(128, 80)
        
#         self.Sigmoid = nn.Sigmoid()
#         self.drop_layer = nn.Dropout(p=0.2)
#         self.drop_layer_2d = nn.Dropout2d(p=0.2)
           
#     def forward(self, x):
#         # first convert the input to float 
#         x = x.squeeze().float()
#         x = x.reshape(x.shape[0], 1, 16, 16)
        
#         conv_layer = nn.Sequential(self.c1,
#                                    nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                    nn.ReLU(),
                                   
#                                    self.c2,
#                                    nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                    nn.ReLU(),
                                   
#                                    self.c3,
#                                    nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                    nn.ReLU(),
                                   
#                                    self.c4,
#                                    nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                    nn.ReLU(),
                                   
#                                    self.c5,
#                                    nn.BatchNorm2d(num_features=512, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.drop_layer_2d)
        
#         phase = conv_layer(x)
#         phase = phase.reshape(phase.shape[0], 1, -1)
        
#         fc_layer = nn.Sequential(self.p11,
#                                  nn.ReLU(),
#                                  self.drop_layer,
                                 
#                                  self.p12,
#                                  nn.ReLU(),
#                                  self.drop_layer,
                                 
#                                  self.p13,
#                                  nn.ReLU(),
#                                  self.drop_layer,
                                 
#                                  self.p2,
#                                  self.p3,
#                                  self.p4,
#                                  self.p5,
#                                  self.Sigmoid)
        
#         phase = fc_layer(phase)

#         return phase.squeeze()




    

# # Phase 6-6-1024
# class PhaseDecoder(nn.Module):
#     def __init__(self, gpu_id):
#         super(PhaseDecoder, self).__init__()
        
#         self.gpu_id = gpu_id

#         # --------------------------FC-LAYER-----------------------------
        
#         self.c1 = nn.Conv2d(1, 32, 3, 1, 1) # N x 32 x 16 x 16
#         self.c2 = nn.Conv2d(32, 64, 3, 1, 1) # N x 64 x 16 x 16
#         self.c3 = nn.Conv2d(64, 128, 3, 1, 1) # N x 128 x 16 x 16
#         self.c4 = nn.Conv2d(128, 256, 4, 2, 1) # N x 256 x 4 x 4
#         self.c5 = nn.Conv2d(256, 512, 4, 2, 1) # N x 512 x 2 x 2
#         self.c6 = nn.Conv2d(512, 1024, 4, 2, 1) # N x 1024 x 2 x 2
        
#         self.p1 = nn.Linear(4096, 2048)
#         self.p2 = nn.Linear(2048, 1024)
#         self.p3 = nn.Linear(1024, 512)
#         self.p4 = nn.Linear(512, 256)
#         self.p5 = nn.Linear(256, 128)
#         self.p6 = nn.Linear(128, 80)
        
#         self.ReLU = nn.ReLU()
           
#     def forward(self, x):
#         # first convert the input to float 
#         x = x.squeeze().float()
#         x = x.reshape(x.shape[0], 1, 16, 16)
        
#         conv_layer = nn.Sequential(self.c1,
#                                    nn.BatchNorm2d(num_features=32, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c2,
#                                    nn.BatchNorm2d(num_features=64, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c3,
#                                    nn.BatchNorm2d(num_features=128, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c4,
#                                    nn.BatchNorm2d(num_features=256, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c5,
#                                    nn.BatchNorm2d(num_features=512, device=self.gpu_id),
#                                    nn.ReLU(),
#                                    self.c6,
#                                    nn.BatchNorm2d(num_features=1024, device=self.gpu_id))
        
#         phase = conv_layer(x)
#         phase = phase.reshape(phase.shape[0], -1)
        
#         fc_layer = nn.Sequential(self.p1,
#                                  self.p2,
#                                  self.p3,
#                                  self.p4,
#                                  self.p5,
#                                  self.p6)
        
#         phase = fc_layer(phase)

#         return phase