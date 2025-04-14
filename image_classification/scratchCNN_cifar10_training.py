import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from neural_networks.scratch_cnn import CNN

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010]
    )
])

training_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                train = True,
                                                transform = all_transforms,
                                                download = True)

test_dataset = torchvision.datasets.CIFAR10(root = "./data",
                                            train = True,
                                            transform = all_transforms,
                                            download = True)

train_loader = torch.utils.data.DataLoader(dataset = training_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)

model = CNN(num_classes)

#Loss function with criterion
criterion = nn.CrossEntropyLoss()

#Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)

total_step = len(train_loader)


#TRAINING

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


#TESTING
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(len(test_loader), 100 * correct/total))