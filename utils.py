import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np

from cnn import CNN

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


class ResNet50ForCIFAR100(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50ForCIFAR100, self).__init__()
        
        # Load the pretrained ResNet50 model
        self.resnet50 = models.resnet50(weights=None)
        
        # Modify the first convolutional layer to use 3x3 kernel with stride 1 for CIFAR-100
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Modify the final fully connected layer to output 100 classes for CIFAR-100
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet50(x)

# Example usage for CIFAR-100
def create_resnet50_for_cifar100():
    model = ResNet50ForCIFAR100(num_classes=100)  # CIFAR-100 has 100 classes
    return model

def subset_train_for_resnet(train, subset_idx, num_epochs = 10, device=torch.device('cuda')):
    lr = 0.1
    batch_size = 512
    criterion = nn.CrossEntropyLoss()
    
    subset = torch.utils.data.Subset(train, subset_idx)

    subsetloader = torch.utils.data.DataLoader(dataset=subset, batch_size=batch_size, shuffle=True)
    
    net = create_resnet50_for_cifar100().to(device)
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