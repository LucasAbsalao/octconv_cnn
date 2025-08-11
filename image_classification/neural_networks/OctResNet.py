import torch
import torch.nn as nn
import torch.functional as F

from neural_networks.octconv import OctaveConv, OctaveConv_BN, OctaveConv_BN_ACT

class OctBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64,
                alpha_in = 0.5, alpha_out = 0.5, norm_layer = None, output = False):
        super(OctBottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        alpha_general = max(alpha_in, alpha_out)
        width = int(planes * (base_width / 64.)) * groups
        print("OctResNet: alpha_in: ", alpha_in, " alpha_out: ", alpha_out)
        print("output: ", output)
        self.conv1 = OctaveConv_BN_ACT(inplanes, width, kernel_size = 1, alpha_in = alpha_in, alpha_out = alpha_out, norm_layer = norm_layer)
        print("output: ", output)
        self.conv2 = OctaveConv_BN_ACT(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                                 alpha_in=0 if output else alpha_general, alpha_out=0 if output else alpha_general)
        print("output: ", output)
        

        self.conv3 = OctaveConv_BN(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                             alpha_in=0 if output else alpha_general, alpha_out=0 if output else alpha_general)
        print("output: ", output)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        #print("tamanhos de ", x_h.size() if isinstance(x_h, torch.Tensor) else "vazio", x_l.size() if isinstance(x_l, torch.Tensor) else "vazio")
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))

        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l
    

class OctResNet(nn.Module):

    def __init__(self, block, layers, num_classes = 1000, zero_init_residual = False,
                 groups = 1, width_per_group = 64, norm_layer = None, num_channels = 3, alpha_in = 0.5, alpha_out = 0.5): #Implementar Alpha_in!!!!!!!!
        super(OctResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size = 7, stride = 2, padding = 3,
                               bias = False)
        self.batch_norm1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer = norm_layer, alpha_in = 0, alpha_out = alpha_out)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2, norm_layer = norm_layer, alpha_in = alpha_in, alpha_out = alpha_out)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2, norm_layer = norm_layer, alpha_in = alpha_in, alpha_out = alpha_out)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2, norm_layer = norm_layer, alpha_in = alpha_in, alpha_out = 0, output = True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677 Não entendi!!!
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,OctBottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride = 1, alpha_in = 0.5, alpha_out = 0.5,
                    norm_layer = None, output = False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        alpha_general = max(alpha_in, alpha_out)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OctaveConv_BN(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride,
                alpha_in = alpha_in, alpha_out = alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion

        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes, groups = self.groups,
                                base_width = self.base_width, norm_layer = norm_layer,
                                alpha_in = 0 if output else alpha_general, alpha_out = 0 if output else alpha_general,
                                output = output))
            

        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h,x_l))
        x_h, x_l = self.layer3((x_h,x_l))
        x_h, x_l = self.layer4((x_h,x_l))

        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

def OctResNet50(pretrained = False, **kwargs):
    model = OctResNet(OctBottleNeck, [3, 4, 6, 3], **kwargs)
    return model

def OctResNet18(pretrained = False, **kwargs):
    model = OctResNet(OctBottleNeck, [2,2,2,2], **kwargs)
    return model

'''
def forward(self, x):
        file=open("log.txt", "a")
        print("OctResNet: \n", file = file)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f"first_x: {x.size()}\n", file = file)

        x_h, x_l = self.layer1(x)
        print(f"x_h1: {x_h.size()}\n", file = file)
        print(f"x_l1: {x_l.size()}\n", file = file)
        x_h, x_l = self.layer2((x_h,x_l))
        print(f"x_h2: {x_h.size()}\n", file = file)
        print(f"x_l2: {x_l.size()}\n", file = file)
        x_h, x_l = self.layer3((x_h,x_l))
        print(f"x_h3: {x_h.size()}\n", file = file)
        print(f"x_l3: {x_l.size()}\n", file = file)
        x_h, x_l = self.layer4((x_h,x_l))
        print(f"x_h4: {x_h.size()}\n", file = file)
        #print(f"x_l4: {x_l.size()}\n", file = file)
        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x'''