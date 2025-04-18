import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np

from cnn import CNN, CNN_CIFAR

def full_train(train, num_epochs=10, device=torch.device('cuda')):
    lr = 0.1
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = CNN(10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    
    return net

def full_train_CIFAR10(train, num_epochs=10, device=torch.device('cuda')):
    lr = 0.1
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = CNN_CIFAR(10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    
    return net

def subset_train(train, subset_idx, num_epochs = 10, device=torch.device('cuda')):
    lr = 0.1
    batch_size = 512
    criterion = nn.CrossEntropyLoss()
    
    subset = torch.utils.data.Subset(train, subset_idx)

    subsetloader = torch.utils.data.DataLoader(dataset=subset, batch_size=batch_size, shuffle=True)
    
    net = CNN(10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in subsetloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    
    return net


def get_resnet18_for_CIFAR100():
    model = models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool to retain spatial resolution
    return model

def full_train_CIFAR_100(train, num_epochs=10, device=torch.device('cuda')):
    lr = 0.1
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = get_resnet18_for_CIFAR100().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f'training on epoch {epoch}')
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    
    return net