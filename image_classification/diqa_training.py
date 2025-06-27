import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from preprocess_images.dataset import get_dataset
from preprocess_images.preprocess import normalize_image
from neural_networks.diqa import DIQA
from train_test_nn.train import train_val_diqa
from train_test_nn.test import validate_model_regression


def execute_diqa(epochs:int):

    live_dataset = get_dataset('live', 'data/Live_IQA_release2/')
    train_dataset, test_dataset = random_split(live_dataset, [0.8,0.2]) 

    trainloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers=2, pin_memory=True)
    valloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers = 2, pin_memory=True)


    torch.cuda.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print(device.index)
    
    model = DIQA().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr = 2 * 10 ** -4, momentum_decay=0.9)
    criterion = torch.nn.MSELoss()

    #print(torch.cuda.memory_summary(device=None, abbreviated=False))

    all_losses = train_val_diqa(model, epochs, criterion, optimizer, device, trainloader, valloader)

    torch.save(model.state_dict(), f"trained_models/model_diqa_{epochs}.pth")
    print("Saved PyTorch Model State to model.pth")

    dict_metrics, mse_total, total = validate_model_regression(model, valloader, device, coefficients=["PLCC","SRCC"])

    return model, dict_metrics

def image_IQA(model, image):
    torch.cuda.init()
    device = torch.device('cpu')
    torch.cuda.empty_cache()

    model = model.to(device)

    totensor = transforms.ToTensor()
    image_tensor = totensor(image)
    image_tensor = normalize_image(image_tensor).unsqueeze(0)

    quality = model(image_tensor.to(device),2)

    return quality
    


if __name__ == '__main__':
    # model = DIQA()
    # model.load_state_dict(torch.load("trained_models/model_diqa_40.pth", weights_only=True))

    # img_path = "../image/arvores.jpg"
    # image = Image.open(img_path).convert('RGB')
    # quality = image_IQA(model, image)
    # print(quality)
    srcc_list, plcc_list = [], []
    for i in range(5):
        model, metrics = execute_diqa(1)
        srcc_list.append(metrics['SRCC'])
        plcc_list.append(metrics['PLCC'])

    plt.plot(srcc_list,color='red', label='SRCC')
    plt.plot(plcc_list, color='blue', label='PLCC')
    plt.axhline(np.mean(srcc_list), color='orange', linestyle='--', label='Média de SRCC')
    plt.axhline(np.mean(srcc_list) + np.std(srcc_list), color='orange', linestyle=':', label='Desvio Padrão de SRCC')
    plt.axhline(np.mean(srcc_list) - np.std(srcc_list), color='orange', linestyle=':')
    plt.axhline(np.mean(plcc_list), color='green', linestyle='--', label='Média de PLCC')
    plt.axhline(np.mean(plcc_list) + np.std(plcc_list), color='green', linestyle=':', label='Desvio Padrão de PLCC')
    plt.axhline(np.mean(plcc_list) - np.std(plcc_list), color='green', linestyle=':')
    plt.legend()
    plt.xlabel("Execuções")
    plt.title("Valores de SRCC e PLCC para várias execuções")
    plt.show()