import torch
import torchvision
import torchvision.transforms as transforms
from neural_networks.resnet import ResNet50
from neural_networks.OctResNet import OctResNet50, OctResNet18

transform_afhq = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.480, 0.448, 0.398], [0.277, 0.269, 0.282])
])

net = ResNet50(10).to('cuda')
net.load_state_dict(torch.load("model_resnet_200.pth", weights_only = True))

val = torchvision.datasets.ImageFolder('data/afhq/val', transform=transform_afhq)

valloader = torch.utils.data.DataLoader(val, batch_size=128,shuffle=False, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
correct = 0
total = 0
with torch.no_grad():
    for (images, labels) in valloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print(f"Accuracy on {len(valloader)} test images: {100*(correct/total)}%")


#ARG argumento para build (Gonça não recomenda: ele não gosta)
#RUN como se fosse no terminal
#WORKDIR cd
#COPY ./nome.cpp / 

# docker run -it name_tag /bin/bash
#htop
# HOST E BRIDGE
#FROM  ubuntu:22.04