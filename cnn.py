import torch
import torchvision
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128)  # Fully connected layer after flattening the feature map
        self.fc2 = nn.Linear(128, num_classes)  # 10 output classes for MNIST digits

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply relu after convolution
        x = torch.max_pool2d(x, 2)    # Max pooling with kernel size 2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.reshape(-1, 7*7*64)  # Flatten the feature map into a vector
        x = torch.relu(self.fc1(x))  # Apply relu activation after the fully connected layer
        x = self.fc2(x)  # Final output layer
        return x
    
class CNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # CIFAR-10 images are 32x32
        self.fc2 = nn.Linear(512, num_classes)  # 10 classes in CIFAR-10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x