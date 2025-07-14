import torch
import torch.utils
import torchvision.models as models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from torch.utils.flop_counter import FlopCounterMode
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from torch_flops import TorchFLOPsByFX

def validate_model(model, valloader, device='cpu', num_classes = 0, labels_name = None, show_cm = False):
    correct = 0
    total = 0
    model.eval()
    all_predicts = []
    all_labels = []

    with torch.no_grad():
        for (images, labels) in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            if show_cm is True:
                all_predicts.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    print(f"Accuracy on {len(valloader) * valloader.batch_size} val images: {100*(correct/total)}%")

    if show_cm is True:
        show_confusion_matrix(np.arange(num_classes), labels_name, all_labels, all_predicts)
        print("\nRelatório de Classificação:\n", classification_report(all_labels, all_predicts))

    return correct, total

def validate_model_regression(model, valloader, device='cpu', loss = 'mse', coefficients = None):
    if loss == 'mse':
        criterion = torch.nn.MSELoss()
    mse_total = 0
    total = 0
    y_pred = []
    y_val = []
    model.eval()
    with torch.no_grad():
        for (images, labels) in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, 2).squeeze(1)


            mse = criterion(outputs.data, labels)
            total += labels.size(0)
            mse_total += mse

            y_val.extend(labels)
            y_pred.extend(outputs.data)

    print(f"Error on {len(valloader) * valloader.batch_size} val images: {torch.sqrt(mse_total)/total}")

    if coefficients is not None:
        y_pred_gpu = torch.stack(y_pred,dim=0)
        y_val_gpu = torch.stack(y_val, dim=0)
        y_pred = y_pred_gpu.cpu()
        y_val = y_val_gpu.cpu()

        dict_metrics = {}

        for coefficient in coefficients:
            if coefficient == 'PLCC':
                plcc = stats.pearsonr(y_pred, y_val)[0]
                dict_metrics['PLCC'] = plcc
                print(f"PLCC: {plcc:.4f}")
            elif coefficient == 'SRCC':
                srcc = stats.spearmanr(y_pred, y_val)[0]
                dict_metrics['SRCC'] = srcc
                print(f"SRCC: {srcc:.4f}")
            elif coefficient == 'KRCC':
                krcc = stats.stats.kenalltau(y_pred,y_val)[0]
                dict_metrics['KRCC'] = krcc
                print(f"KRCC: {krcc:.4f}")
        return dict_metrics, mse_total, total

    return mse_total, total

def show_confusion_matrix(labels, labels_name, expected, predicted):
    cm = confusion_matrix(y_true = expected, y_pred = predicted, labels = labels)
    if len(labels) > 50: 
        print(cm)
    else:
        cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels_name)
        cm_disp.plot()
        plt.show()

def get_flops_counter_mode(model, input, device = 'cpu'):
    istrain = model.training
    model.eval()

    input = input.to(device)
    print(model(input))
    
    if not isinstance(input, torch.Tensor):
        print("Gerando números aleatórios para calcular a quantidade de flops")
        input = torch.randn(input)

    print("Tipo do tensor para calcular o flop: ", input.type())
    flop_counter = FlopCounterMode(mods = model, display = False, depth = None)
    with flop_counter:
        print(model(input))

    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

def get_flops_fvcore(model, input, device = 'cpu'):
    istrain = model.training
    model.eval()

    input = input.to(device)
    print(model(input))
    
    if not isinstance(input, torch.Tensor):
        print("Gerando números aleatórios para calcular a quantidade de flops")
        input = torch.randn(input)

    print("Tipo do tensor para calcular o flop: ", input.type())
    
    flops = FlopCountAnalysis(model, input)
    if istrain:
        model.train()
    return flops.total()

def get_flops_ptflops(model, input, device = 'cpu'):

    istrain = model.training
    model.eval()

    input = input.to(device)
    print(model(input))
    
    if not isinstance(input, torch.Tensor):
        print("Gerando números aleatórios para calcular a quantidade de flops")
        input = torch.randn(input)

    print("Tipo do tensor para calcular o flop: ", input.type())
    print("Tamanho do tensor para calcular o flop: ", tuple(input.size())[1:])
    
    with torch.cuda.device(device.index if device.index is not None else device):
        flops, params = get_model_complexity_info(model, tuple(input.size()[1:]), as_strings = True)
        print(f'FLOPs: {flops}')
        print(f"Params {params}")
    if istrain:
        model.train()
    return flops

def get_flops_torch_flops(model, input, device = 'cpu'):

    istrain = model.training
    model.eval()

    input = input.to(device)
    print(model(input))
    
    if not isinstance(input, torch.Tensor):
        print("Gerando números aleatórios para calcular a quantidade de flops")
        input = torch.randn(input)

    print("Tipo do tensor para calcular o flop: ", input.type())
    print("Tamanho do tensor para calcular o flop: ", tuple(input.size())[1:])
    
    with torch.no_grad():
        model(input)

    flops_counter = TorchFLOPsByFX(model)
    flops_counter.propagate(input)

    result_table = flops_counter.print_result_table()
    total_flops = flops_counter.print_total_flops(show=True)
    total_time = flops_counter.print_total_time()
    max_memory = flops_counter.print_max_memory()

    if istrain:
        model.train()
    return total_flops
