import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from .test import validate_model, validate_model_regression
from tqdm import tqdm

def train_model(model, epochs, criterion, optimizer, scheduler, device, trainloader):
    all_losses = []
    model.train()
    for epoch in range(epochs):
        losses = []
        running_loss = 0
        for i, inp in enumerate(tqdm(trainloader)):
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
        if scheduler is not None:
            scheduler.step(avg_loss)
        all_losses.append(avg_loss)
                
    print('Training Done')
    plot_loss(all_losses, epochs)
    return all_losses

def plot_loss(all_losses, title=None):
    fig = plt.figure(figsize=(15,15))
    plt.plot(all_losses, marker = 'o')
    if title is not None:
        plt.title(title)
    plt.show()

def train_val_model(model, epochs, criterion, optimizer, scheduler, device, trainloader, valloader):
    all_losses = []
    val_accuracy = []
    for epoch in range(epochs):
        model.train()
        losses = []
        running_loss = 0
        for i, inp in enumerate(tqdm(trainloader)):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
        running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        if scheduler is not None:
            scheduler.step(avg_loss)
        all_losses.append(avg_loss)

        correct, total = validate_model(model, valloader, device)
        val_accuracy.append(100*(correct/total))
                
    print('Training Done')
    plot_loss(all_losses, 'Loss por época')
    plot_loss(val_accuracy, "Accuracy por época")
    return all_losses

def train_val_diqa(model, epochs, criterion, optimizer, device, trainloader, valloader):
    all_losses = []
    val_accuracy = []
    for epoch in range(epochs):
        model.train()
        losses = []
        running_loss = 0
        for i, inp in enumerate(tqdm(trainloader)):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            # print(type(inputs), type(labels))
            # print(device)
            optimizer.zero_grad()
        
            outputs = model(inputs,2).squeeze(1)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
        running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        all_losses.append(avg_loss)

        correct, total = validate_model_regression(model, valloader, device)
        val_accuracy.append(100*(correct/total))
                
    print('Training Done')
    plot_loss(all_losses, 'Loss por época')
    plot_loss(val_accuracy, "Accuracy por época")
    return all_losses