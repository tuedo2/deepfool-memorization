import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from cnn import CNN

def subset_train(train, subset_idx, num_epochs = 10, device=torch.device('cuda')):
    lr = 0.001
    batch_size = 256
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