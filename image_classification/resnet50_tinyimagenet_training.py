import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from neural_networks.resnet import Bottleneck, ResNet50, ResNet
from neural_networks.OctResNet import OctResNet50, OctResNet18
from train_test_nn.train import train_model, train_val_model
from train_test_nn.test import validate_model

# import gc
# del variables
# gc.collect()


transform_afhq = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


train = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', transform=transform_afhq)
trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=2)

val = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', transform=transform_afhq)
valloader = torch.utils.data.DataLoader(val, batch_size=64,shuffle=False, num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

net = OctResNet50(num_classes = 200).to(device)

criterion = nn.CrossEntropyLoss()
print(type(criterion))
optimizer = optim.Adam(net.parameters(), lr=0.001)
print(type(optimizer))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
print(type(scheduler))

print(torch.cuda.memory_summary(device=None, abbreviated=False))

EPOCHS = 1
all_losses = train_val_model(net, EPOCHS, criterion, optimizer, scheduler, device, trainloader, valloader)

torch.save(net.state_dict(), f"trained_models/model_{EPOCHS}.pth")
print("Saved PyTorch Model State to model.pth")

correct, total = validate_model(net, valloader, device, num_classes = 200, show_cm = True)