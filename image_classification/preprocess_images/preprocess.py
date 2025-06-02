import torch 
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

def get_gaussian_kernel(kernel_size: int, sigma: float, dtype = torch.float32):
    kernel_range = torch.arange(-kernel_size/2, kernel_size/2)
    x, y = torch.meshgrid(kernel_range, kernel_range, indexing='ij')

    kernel = 1 / (2 * math.pi * sigma ** 2) * torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel.to(torch.float32)
    return kernel/torch.sum(kernel)

def normalize_image(image):
    kernel = get_gaussian_kernel(17, 4)
    kernel_3d = kernel.unsqueeze(0).repeat(3,1)
    gaussian_blur = F.conv2d(image, weight = kernel_3d, padding = 'same')
    show_img(gaussian_blur)

def show_img(image):
    img = torch.permute(torch.squeeze(image), (1,2,0))
    plt.imshow(img, cmap = 'gray')
    plt.show()

teste = cv2.imread('bikes.bmp')
teste = cv2.cvtColor(teste, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToTensor()
])
teste_tensor = transform(teste)
teste_tensor = torch.unsqueeze(teste_tensor,0)
normalize_image(teste_tensor)