import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from preprocess_images.dataset import get_dataset
from preprocess_images.preprocess import normalize_image
from neural_networks.diqa import DIQA
from neural_networks.octdiqa import OctDIQA
from train_test_nn.train import train_val_diqa
from train_test_nn.test import validate_model_regression


def execute_diqa(epochs:int, dataset_name:str, batch_size:int=1):
    #O batch size não influencia o live, pois, como as imagens possuem tamanho diferente, é mais difícil de fazer os batchs (teria que mexer na função sampler)
    if dataset_name=='live':
        dataset = get_dataset('live', 'data/Live_IQA_release2/')
        batch_size = 1
    elif dataset_name=='koniq-10k':
        dataset = get_dataset('koniq-10k', 'data/KonIQ-10k/', 'data/KonIQ-10k/512x384/')
    else:
        raise Exception("não tem dataset")
    
    train_dataset, test_dataset = random_split(dataset, [0.8,0.2]) 

    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=2, pin_memory=True)
    valloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory=True)


    torch.cuda.init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    model = OctDIQA().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr = 2 * 10 ** -4, momentum_decay=0.9)
    criterion = torch.nn.MSELoss()

    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    all_losses = train_val_diqa(model, epochs, criterion, optimizer, device, trainloader, valloader)

    dict_metrics, mse_total, total = validate_model_regression(model, valloader, device, coefficients=["PLCC","SRCC"])

    return model, dict_metrics, all_losses

def execute_diqa_koniq(epochs:int):

    live_dataset = get_dataset('koniq-10k', 'data/KonIQ-10k/', 'data/KonIQ-10k/512x384/')
    train_dataset, test_dataset = random_split(live_dataset, [0.8,0.2]) 

    trainloader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers=2, pin_memory=True)
    valloader = DataLoader(test_dataset, batch_size = 4, shuffle = True, num_workers = 2, pin_memory=True)


    torch.cuda.init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    model = DIQA().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr = 2 * 10 ** -4, momentum_decay=0.9)
    criterion = torch.nn.MSELoss()

    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    all_losses = train_val_diqa(model, epochs, criterion, optimizer, device, trainloader, valloader)

    dict_metrics, mse_total, total = validate_model_regression(model, valloader, device, coefficients=["PLCC","SRCC"])

    return model, dict_metrics

def image_IQA(model, image_path):
    image = Image.open(image_path).convert('RGB')

    torch.cuda.init()
    device = torch.device('cpu')
    torch.cuda.empty_cache()

    model = model.to(device)

    totensor = transforms.ToTensor()
    image_tensor = totensor(image)
    image_tensor = normalize_image(image_tensor).unsqueeze(0)

    quality = model(image_tensor.to(device),2)

    return quality
    
def plot_srcc_list(srcc_list:list, plcc_list:list, name:str):
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
    plt.savefig(f"stats/graphs/{name}.png",bbox_inches = 'tight')

def write_metrics_txt(srcc_list:list, plcc_list:list, all_losses:list, name:str):
    with open(f"stats/{name}.txt", "w") as f:
        f.write(f"Média do SRCC - {np.mean(srcc_list):.4f} e desvio padrão - {np.std(srcc_list):.4f}\n")
        f.write(f"Valores de SRCC - {srcc_list}\n\n")
        f.write(f"Média do PLCC - {np.mean(plcc_list):.4f} e desvio padrão de {np.std(plcc_list):.4f}\n")
        f.write(f"Valores de PLCC - {plcc_list}\n")
        for losses in all_losses:
            f.write(f"Evolução da loss: {losses}\n")

if __name__ == '__main__':
    # model = DIQA()
    # model.load_state_dict(torch.load("trained_models/model_diqa_40.pth", weights_only=True))

    # img_path = "../image/arvores.jpg"
    # quality = image_IQA(model, image)
    # print(quality)
    name = "DIQA_koniq"
    epochs = 1
    executions = 1

    srcc_list, plcc_list, loss_list = [], [], []
    for i in range(executions):
        model, metrics, all_losses = execute_diqa(epochs, 'koniq-10k', 32)
        srcc_list.append(metrics['SRCC'])
        plcc_list.append(metrics['PLCC'])
        loss_list.append(all_losses)
        if metrics['SRCC'] >= max(srcc_list):
            torch.save(model.state_dict(), f"trained_models/model_{name}_{epochs}.pth")
            print(f"Saved PyTorch Model State to model on execution {i}.pth")

    plot_srcc_list(srcc_list, plcc_list, name)

    write_metrics_txt(srcc_list, plcc_list, name)