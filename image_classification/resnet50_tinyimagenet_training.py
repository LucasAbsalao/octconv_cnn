import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from neural_networks.resnet import ResNet50
from neural_networks.OctResNet import OctResNet50, OctResNet18
from train_test_nn.train import train_model, train_val_model
from train_test_nn.test import validate_model, get_flops_counter_mode, get_flops_fvcore, get_flops_ptflops, get_flops_torch_flops


def execute_model(type_model:str, epochs:int, dataset: str, num_classes: int, flops_mode: str, alpha: float = 0.5):

    # -------------------------------------- Image Processing -----------------------------------------------
    transform_afhq = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if dataset == "afhq":
        path_dataset = 'data/afhq'
    elif dataset == "tiny-imagenet":
        path_dataset = 'data/tiny-imagenet-200'

    train = torchvision.datasets.ImageFolder(path_dataset+'/train', transform=transform_afhq)
    trainloader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, num_workers=2)

    val = torchvision.datasets.ImageFolder(path_dataset+'/val', transform=transform_afhq)
    valloader = torch.utils.data.DataLoader(val, batch_size=16,shuffle=False, num_workers=2)


    # -------------------------------------- Treinamento -----------------------------------------------
    torch.cuda.init()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device used: ", device)
    torch.cuda.empty_cache()

    if type_model == "ResNet50":
        model = ResNet50(num_classes = num_classes).to(device)
    elif type_model == "OctResNet50":
        model = OctResNet50(num_classes = num_classes, alpha_in = alpha, alpha_out = alpha).to(device)
    else:
        print("Modelo não suportado")
        return 0 

    criterion = nn.CrossEntropyLoss()
    print(type(criterion))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(type(optimizer))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    print(type(scheduler))

    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    all_losses = train_val_model(model, epochs, criterion, optimizer, scheduler, device, trainloader, valloader)


    # -------------------------------------- Salvar e Validar  -----------------------------------------------
    torch.save(model.state_dict(), f"trained_models/model_{epochs}.pth")
    print("Saved PyTorch Model State to model.pth")

    correct, total = validate_model(model, valloader, device, num_classes = 3, labels_name = ['Cat', 'Dog', 'Wild'], show_cm = True)



    # -------------------------------------- FLOPs -----------------------------------------------
    valloader_2 = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False, num_workers=2)
    first_input, _ = next(iter(valloader_2))

    match flops_mode:
        case "counter_mode":
            flops = get_flops_counter_mode(model, first_input, device)
        case "fvcore":
            flops = get_flops_fvcore(model, first_input, device)
        case "ptflops":
            flops = get_flops_ptflops(model, first_input, device)
        case "torch_flops":
            flops = get_flops_torch_flops(model, first_input, device)
        case _:
            return model
    
    print("A quantidade de flops foi de ", flops, " flops para um tensor de ", first_input.size())

    with open("flops.txt", 'a') as file:
        file.write(f"{flops_mode} - \n")
        file.write(f"\t{type_model} - A quantidade de flops foi de {flops} flops para um tensor de {first_input.size()}" + f" e alpha {alpha}" if type_model=='OctResNet50' else '' + "\n")

    return model

model = execute_model(type_model = "OctResNet50", epochs = 10, dataset = "afhq", num_classes = 3, flops_mode = "ptflops", alpha = 0.75)
print("Script finalizado")