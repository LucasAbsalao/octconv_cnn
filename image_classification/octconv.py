# Code from d-li14 available at https://github.com/d-li14/octconv.pytorch.git

import torch
import torch.nn as nn
import math

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in = 0.5, alpha_out = 0.5, stride = 1, padding = 0, dilation = 1,
                 groups = 1, bias = False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size = (2, 2), stride = 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        assert stride == 1 or stride == 2, "Stride should be 1 or 2"
        self.stride = stride

        self.is_dw = groups == in_channels

        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alpha should be in the inerval from 0 to 1"
        self.alpha_in, self.alpha_out = alpha_in, alpha_out

        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels), kernel_size = kernel_size, stride = 1, 
                                  padding = padding, dilation = dilation, groups=math.ceil(alpha_in * groups), bias = bias)
        
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels), kernel_size = kernel_size, stride = 1, 
                                  padding = padding, dilation = dilation, groups=groups, bias = bias)
        
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels), kernel_size = kernel_size, stride = 1, 
                                  padding = padding, dilation = dilation, groups=groups, bias = bias)
        
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels), kernel_size = kernel_size, stride = 1, 
                                  padding = padding, dilation = dilation, groups=math.ceil(groups - alpha_in * groups), bias = bias)
        
    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h

        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None

        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h