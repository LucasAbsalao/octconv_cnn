import torch
import torch.nn as nn
from .octconv import OctaveConv, OctaveConv_ACT

class OctDIQA1(nn.Module):

    def __init__(self, alpha, conv_vector:list):
        super(OctDIQA, self).__init__()
        self.conv_vector = conv_vector

        if conv_vector[0] == 0:
            self.conv1 = nn.Conv2d(in_channels = 1,
                                   out_channels = 48,
                                   kernel_size = 3,
                                   stride = 1,
                                   padding = 1)
        else:
            self.conv1 = OctaveConv(in_channels = 1)


class OctDIQA(nn.Module):
    def __init__(self, alpha=0.125):
        super(OctDIQA,self).__init__()
        self.alpha = alpha 
        self.conv1 = OctaveConv_ACT(in_channels = 1, 
                            out_channels = 48, 
                            kernel_size = 3,
                            alpha_in = 0,
                            alpha_out = self.alpha, 
                            stride = 1, 
                            padding = 1,
                            deactivate_dw = True)

        self.conv2 = OctaveConv_ACT(in_channels = 48,
                            out_channels = 48,
                            kernel_size = 3,
                            alpha_in = self.alpha,
                            alpha_out = self.alpha,
                            stride = 2, 
                            padding = 1)

        self.conv3 = OctaveConv_ACT(in_channels = 48,
                            out_channels = 64,
                            kernel_size = 3,
                            alpha_in = self.alpha,
                            alpha_out = self.alpha,
                            stride = 1,
                            padding = 1)

        self.conv4 = OctaveConv_ACT(in_channels = 64,
                               out_channels = 64,
                               kernel_size = 3,
                               alpha_in = self.alpha,
                               alpha_out = self.alpha,
                               stride = 2,
                               padding = 1)

        self.conv5 = OctaveConv_ACT(in_channels = 64,
                               out_channels = 64,
                               kernel_size = 3,
                               alpha_in = self.alpha,
                               alpha_out = self.alpha,
                               stride = 1,
                               padding = 1)

        self.conv6 = OctaveConv_ACT(in_channels = 64,
                               out_channels = 64,
                               kernel_size = 3,
                               alpha_in = self.alpha,
                               alpha_out = self.alpha,
                               stride = 1,
                               padding = 1)

        self.conv7 = OctaveConv_ACT(in_channels = 64,
                               out_channels = 128,
                               kernel_size = 3,
                               alpha_in = self.alpha,
                               alpha_out = self.alpha,
                               stride = 1,
                               padding = 1)

        self.conv8 = OctaveConv_ACT(in_channels = 128,
                               out_channels = 128,
                               kernel_size = 3,
                               alpha_in = self.alpha,
                               alpha_out = 0,
                               stride = 1,
                               padding = 1)

        self.conv9 = OctaveConv_ACT(in_channels = 128,
                               out_channels = 1,
                               kernel_size = 1,
                               alpha_in = 0,
                               alpha_out = 0,
                               stride = 2,
                               padding = 1)

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, mode=2):
        x_h, x_l = self.conv1(x)
        #print("conv1: ", x_h.size(), x_l.size() if x_l is not None else None)
        x_h, x_l = self.conv2((x_h,x_l))
        #print("conv2: ", x_h.size(), x_l.size() if x_l is not None else None)
        x_h, x_l = self.conv3((x_h,x_l))
        #print("conv3: ", x_h.size(), x_l.size() if x_l is not None else None)
        x_h, x_l = self.conv4((x_h,x_l))
        #print("conv4: ", x_h.size(), x_l.size() if x_l is not None else None)
        x_h, x_l = self.conv5((x_h,x_l))
        #print("conv5: ", x_h.size(), x_l.size() if x_l is not None else None)
        x_h, x_l = self.conv6((x_h,x_l))
        #print("conv6: ", x_h.size(), x_l.size() if x_l is not None else None)
        x_h, x_l = self.conv7((x_h,x_l))
        #print("conv7: ", x_h.size(), x_l.size() if x_l is not None else None)
        x_h_8, x_l_8 = self.conv8((x_h, x_l))
        #print("conv8: ", x_h_8.size(), x_l_8.size() if x_l_8 is not None else None)
        #print()
        if mode == 1:
            e = self.conv9(x_h_8)
            return e
        
        else:
            s =self.global_average_pooling(x_h_8)
            s = s.squeeze(3).squeeze(2)
            s = self.relu(self.fc1(s))
            s = self.fc2(s)
            return s