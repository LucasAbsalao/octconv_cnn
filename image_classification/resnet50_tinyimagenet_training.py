import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neural_networks.resnet import Bottleneck, ResNet50, ResNet
from train.train import train_model


transform_afhq = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.480, 0.448, 0.398], [0.277, 0.269, 0.282])
])


train = torchvision.datasets.ImageFolder('data/afhq/train', transform=transform_afhq)

trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

val = torchvision.datasets.ImageFolder('data/afhq/val', transform=transform_afhq)

valloader = torch.utils.data.DataLoader(val, batch_size=128,shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = ResNet50(3).to(device)

criterion = nn.CrossEntropyLoss()
print(type(criterion))
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
print(type(optimizer))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
print(type(scheduler))

EPOCHS = 10

all_losses = train_model(net, EPOCHS, criterion, optimizer, scheduler, device, trainloader)

torch.save(net.state_dict(), f"model_{EPOCHS}.pth")
print("Saved PyTorch Model State to model.pth")

correct = 0
total = 0
with torch.no_grad():
    for (images, labels) in valloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()


print(f"Accuracy on {len(valloader)} val images: {100*(correct/total)}%")