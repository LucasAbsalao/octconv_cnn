import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_model(model, epochs, criterion, optimizer, scheduler, device, trainloader):
    all_losses = []
    for epoch in range(epochs):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i%100 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
        all_losses.append(avg_loss)
                
    print('Training Done')
    return all_losses