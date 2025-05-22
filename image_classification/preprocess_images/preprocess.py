import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import math

def get_gaussian_kernel(kernel_size: int, sigma: float, dtype = torch.float32):
    kernel_range = torch.arange(-kernel_size/2, kernel_size/2)
    x, y = torch.meshgrid(kernel_range, kernel_range, indexing='ij')

    kernel = 1 / (2 * math.pi * sigma ** 2) * torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel/torch.sum(kernel)

def normalize_image(image):
    kernel = get_gaussian_kernel(17, 4)
    plt.imshow(kernel, cmap = 'gray')
    plt.show()


normalize_image(9)