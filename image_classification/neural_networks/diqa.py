import torch
import torch.nn as nn


class DIQA(nn.Module):

    def __init__(self):
        super(DIQA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                            out_channels = 48, 
                            kernel_size = 3, 
                            stride = 1, 
                            padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 48,
                            out_channels = 48,
                            kernel_size = 3,
                            stride = 2, 
                            padding = 1)

        self.conv3 = nn.Conv2d(in_channels = 48,
                            out_channels = 64,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1)

        self.conv4 = nn.Conv2d(in_channels = 64,
                               out_channels = 64,
                               kernel_size = 3,
                               stride = 2,
                               padding = 1)

        self.conv5 = nn.Conv2d(in_channels = 64,
                               out_channels = 64,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)

        self.conv6 = nn.Conv2d(in_channels = 64,
                               out_channels = 64,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)

        self.conv7 = nn.Conv2d(in_channels = 64,
                               out_channels = 128,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)

        self.conv8 = nn.Conv2d(in_channels = 128,
                               out_channels = 128,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)

        self.conv9 = nn.Conv2d(in_channels = 128,
                               out_channels = 1,
                               kernel_size = 1,
                               stride = 2,
                               padding = 1)

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, mode):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        result_conv8 = self.relu(self.conv8(x))


        if mode == 1:
            e = self.conv9(result_conv8)
            return e
        else:
            s = self.global_average_pooling(result_conv8)
            s = s.squeeze(2).squeeze(3)
            s = self.relu(self.fc1(s)) #Autoria própria
            s = self.fc2(s)
            return s

            