import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from neural_networks.resnet import Bottleneck, ResNet50, ResNet
from neural_networks.OctResNet import OctResNet50, OctResNet18
from train_test_nn.train import train_model, train_val_model
from train_test_nn.test import validate_model, get_flops, get_flops_fvcore, get_flops_ptflops



transform_afhq = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


train = torchvision.datasets.ImageFolder('data/afhq/train', transform=transform_afhq)
trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=2)

val = torchvision.datasets.ImageFolder('data/afhq/val', transform=transform_afhq)
valloader = torch.utils.data.DataLoader(val, batch_size=64,shuffle=False, num_workers=2)

torch.cuda.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()


model = OctResNet50(num_classes = 3).to(device)

first_tensor, _ = next(iter(trainloader))
size_tensor = first_tensor.size()
print(size_tensor)

criterion = nn.CrossEntropyLoss()
print(type(criterion))
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(type(optimizer))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
print(type(scheduler))

print(torch.cuda.memory_summary(device=None, abbreviated=False))

EPOCHS = 1
all_losses = train_val_model(model, EPOCHS, criterion, optimizer, scheduler, device, trainloader, valloader)

torch.save(model.state_dict(), f"trained_models/model_{EPOCHS}.pth")
print("Saved PyTorch Model State to model.pth")

correct, total = validate_model(model, valloader, device, num_classes = 3, labels_name = ['Cat', 'Dog', 'Wild'], show_cm = True)

valloader_2 = torch.utils.data.DataLoader(val, batch_size=1, shuffle=False, num_workers=2)
first_input, _ = next(iter(valloader_2))
flops = get_flops_ptflops(model, first_input, device)
print("A quantidade de flops foi de ", flops, " flops para um tensor de ", first_input.size())