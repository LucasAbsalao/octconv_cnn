import torch
from preprocess_images.dataset import get_dataset, custom_collate_fn
from torch.utils.data import random_split, DataLoader
from neural_networks.diqa import DIQA
from train_test_nn.train import train_val_diqa
from train_test_nn.test import validate_model_regression


def execute_diqa(epochs:int):

    live_dataset = get_dataset('live', 'data/Live_IQA_release2/')
    train_dataset, test_dataset = random_split(live_dataset, [0.8,0.2]) 

    trainloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers=2)
    valloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers = 2)


    torch.cuda.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    model = DIQA().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr = 2 * 10 ** -4)
    criterion = torch.nn.MSELoss()

    #print(torch.cuda.memory_summary(device=None, abbreviated=False))

    all_losses = train_val_diqa(model, epochs, criterion, optimizer, device, trainloader, valloader)

    torch.save(model.state_dict(), f"trained_models/model_diqa_{epochs}.pth")
    print("Saved PyTorch Model State to model.pth")

    mse_total, total = validate_model_regression(model, valloader,device,)
    return model

model = execute_diqa(1)