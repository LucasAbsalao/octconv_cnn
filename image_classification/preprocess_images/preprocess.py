import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

def get_gaussian_kernel(kernel_size: int, sigma: float, dtype = torch.float32):
    kernel_range = torch.arange(-kernel_size/2, kernel_size/2)
    x, y = torch.meshgrid(kernel_range, kernel_range, indexing='ij')

    kernel = 1 / (2 * math.pi * sigma ** 2) * torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel.to(torch.float32)

    return kernel/torch.sum(kernel)

def downupsample(img, ratio:float=4):
    c, h, w = img.shape
    h_new, w_new = h//ratio, w//ratio


    downsample = transforms.Resize((h_new, w_new), antialias=True)
    upsample = transforms.Resize((h,w), antialias=True)
    down = downsample(img)
    up = upsample(down)
    return up

def normalize_image(image):
    #print(image.size())
    image_gray = transforms.functional.rgb_to_grayscale(image)
    print(image_gray.size())
    kernel = get_gaussian_kernel(17, 7/6)
    gaussian_blur = F.conv2d(image_gray, weight = kernel.unsqueeze(0).unsqueeze(0), padding = 'same')
    #print(gaussian_blur.size())
    resized = downupsample(gaussian_blur)
    preprocessed_image = image_gray - resized
    # show_img(image_gray)
    # show_img(resized)
    # show_img(preprocessed_image)
    return preprocessed_image

def show_img(image):
    if len(image.size()) == 4:

        img = image[0]
    else:
        img = image
    print(img.size())
    img = torch.permute(img, (1,2,0))
    plt.imshow(img, cmap = 'gray')
    plt.show()

# teste = cv2.imread('bikes.bmp')
# teste = cv2.cvtColor(teste, cv2.COLOR_BGR2RGB)

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# teste_tensor = transform(teste)
# teste_tensor = torch.unsqueeze(teste_tensor,0)
# image = normalize_image(teste_tensor)