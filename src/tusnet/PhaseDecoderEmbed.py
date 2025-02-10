## ------------------------------------------------------------
## IMPORTS
## ------------------------------------------------------------

import torch
import torch.nn as nn
from LSTMConvCell import *

# Phase Embed Dynamic
class PhaseDecoder(nn.Module):
    def __init__(self, gpu_id, ch_mult):
        super(PhaseDecoder, self).__init__()

        self.gpu_id = gpu_id
        self.ch_mult = ch_mult  # channel multiplier

        # primary convolutions on outputs of blocks 1, 2, 3, and 4 (reduced embedding)
        self.blk1 = nn.Conv2d(in_channels=1, out_channels=16 * self.ch_mult, kernel_size=4,
                     stride=2, padding=1)
        self.blk2 = nn.Conv2d(in_channels=1, out_channels=16 * self.ch_mult, kernel_size=3,
                     stride=1, padding=1)
        self.blk3 = nn.Conv2d(in_channels=1, out_channels=8 * self.ch_mult, kernel_size=3,
                     stride=1, padding=1)
        self.blk4 = nn.Conv2d(in_channels=1, out_channels=4 * self.ch_mult, kernel_size=3,
                     stride=1, padding=1)
        self.blk5 = nn.Conv2d(in_channels=1, out_channels=2 * self.ch_mult, kernel_size=3,
                     stride=1, padding=1)

        # secondary convolutions applied to concatenated tensors:
        self.blk2_second = nn.Conv2d(in_channels=16 * self.ch_mult * 2, out_channels=8 * self.ch_mult, kernel_size=4,
                     stride=2, padding=1)
        self.blk3_second = nn.Conv2d(in_channels=8 * self.ch_mult * 2, out_channels=4 * self.ch_mult, kernel_size=4,
                     stride=2, padding=1)
        self.blk4_second = nn.Conv2d(in_channels=4 * self.ch_mult * 2, out_channels=2 * self.ch_mult, kernel_size=4,
                     stride=2, padding=1)

        # tertiary convolutions applied to the final output
        self.c1 = nn.Conv2d(in_channels=4 * self.ch_mult, out_channels=32, kernel_size=4,
                     stride=2, padding=0)
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                     stride=1, padding=0)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                     stride=1, padding=0)
        self.c4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                     stride=1, padding=0)

        # fully connected layers
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=80)

    def forward(self, encoded):

        # unwrap encodings
        x1 = encoded[4].float()
        x2 = encoded[3].float()
        x3 = encoded[2].float()
        x4 = encoded[1].float()
        x5 = encoded[0].float()

        # first block of convolutions
        primary_conv_blk1 = nn.Sequential(self.blk1,
                                         nn.BatchNorm2d(num_features=16 * self.ch_mult, device=self.gpu_id),
                                         nn.ReLU(),
                                         nn.Dropout(0.2))

        primary_conv_blk2 = nn.Sequential(self.blk2,
                                         nn.BatchNorm2d(num_features=16 * self.ch_mult, device=self.gpu_id),
                                         nn.ReLU(),
                                         nn.Dropout(0.2))

        primary_conv_blk3 = nn.Sequential(self.blk3,
                                         nn.BatchNorm2d(num_features=8 * self.ch_mult, device=self.gpu_id),
                                         nn.ReLU(),
                                         nn.Dropout(0.2))

        primary_conv_blk4 = nn.Sequential(self.blk4,
                                         nn.BatchNorm2d(num_features=4 * self.ch_mult, device=self.gpu_id),
                                         nn.ReLU(),
                                         nn.Dropout(0.2))

        primary_conv_blk5 = nn.Sequential(self.blk5,
                                         nn.BatchNorm2d(num_features=2 * self.ch_mult, device=self.gpu_id),
                                         nn.ReLU(),
                                         nn.Dropout(0.2))

        # second block of convolutions
        secondary_conv_blk2 = nn.Sequential(self.blk2_second,
                                           nn.BatchNorm2d(num_features=8 * self.ch_mult, device=self.gpu_id),
                                           nn.ReLU(),
                                           nn.Dropout(0.2))

        secondary_conv_blk3 = nn.Sequential(self.blk3_second,
                                           nn.BatchNorm2d(num_features=4 * self.ch_mult, device=self.gpu_id),
                                           nn.ReLU(),
                                           nn.Dropout(0.2))

        secondary_conv_blk4 = nn.Sequential(self.blk4_second,
                                           nn.BatchNorm2d(num_features=2 * self.ch_mult, device=self.gpu_id),
                                           nn.ReLU(),
                                           nn.Dropout(0.2))

        # tertiary block of convolutions
        tertiary_conv_layers = nn.Sequential(self.c1,
                                             nn.BatchNorm2d(num_features=32, device=self.gpu_id),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),

                                             self.c2,
                                             nn.BatchNorm2d(num_features=64, device=self.gpu_id),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),

                                             self.c3,
                                             nn.BatchNorm2d(num_features=128, device=self.gpu_id),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),

                                             self.c4,
                                             nn.BatchNorm2d(num_features=256, device=self.gpu_id),
                                             nn.ReLU(),
                                             nn.Dropout(0.2))

        fully_connected_layers = nn.Sequential(self.fc1, self.fc2)


        x1_out = primary_conv_blk1(x1)                            # (N, 16, 128, 128)
        x2_out = primary_conv_blk2(x2)                            # (N, 16, 128, 128)
        x3_out = primary_conv_blk3(x3)                            # (N, 8, 64, 64)
        x4_out = primary_conv_blk4(x4)                            # (N, 4, 32, 32)
        x5_out = primary_conv_blk5(x5)                            # (N, 2, 16, 16)

        x12_cat = torch.concat((x1_out, x2_out), dim=1)           # (N, 32, 128, 128)
        x12_catconv = secondary_conv_blk2(x12_cat)                # (N, 8, 64, 64)
        x123_cat = torch.concat((x12_catconv, x3_out), dim=1)     # (N, 16, 64, 64)
        x123_catconv = secondary_conv_blk3(x123_cat)              # (N, 4, 32, 32)
        x1234_cat = torch.concat((x123_catconv, x4_out), dim=1)   # (N, 8, 32, 32)
        x1234_catconv = secondary_conv_blk4(x1234_cat)            # (N, 2, 16, 16)
        x12345_cat = torch.concat((x1234_catconv, x5_out), dim=1) # (N, 4, 16, 16)
        conv_out = tertiary_conv_layers(x12345_cat)               # (N, 256, 1, 1)

        # # uncomment below to see the state dimensions
        # print('primary convolutional block dimensions')
        # for j in [x1_out, x2_out, x3_out, x4_out, x5_out]:
        #     print(f'size: {j.shape}')
        # print('secondary concatenation block dimensions')
        # for j in [x12_cat, x123_cat, x1234_cat, x12345_cat]:
        #     print(f'size: {j.shape}')
        # print('secondary convlotional block dimensions')
        # for j in [x12_catconv, x123_catconv, x1234_catconv, conv_out]:
        #     print(f'size: {j.shape}')

        fc_out = fully_connected_layers(conv_out.squeeze())       # (N, 80)

        phase = fc_out.squeeze()

        return phase