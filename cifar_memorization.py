import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import subset_train_for_resnet

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
cifar = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=False)
trainloader = torch.utils.data.DataLoader(dataset=cifar, batch_size=512, shuffle=False)

def get_correctness_for_model(model):
    model.eval()
    trainset_correctness = []
    with torch.no_grad():
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            correctness = (predicted == targets).float()
            trainset_correctness.append(correctness)
    
    trainset_correctness = torch.cat(trainset_correctness).cpu()
    return trainset_correctness

num_models = 1000
subset_ratio = 0.7

masks = []
correctnesses = []

for i in range(num_models):
    perm = torch.randperm(len(cifar)).numpy()
    
    subset_idx = perm[(len(cifar) * subset_ratio):]
    
    mask = np.zeros(len(cifar), dtype=bool)
    mask[subset_idx] = True
    masks.append(mask)
    
    net = subset_train_for_resnet(cifar, subset_idx, num_epochs=70, device=device)

    correctness = get_correctness_for_model(net)
    correctnesses.append(correctness)

mask_matrix = np.vstack([mask for mask in masks])
correctness_matrix = np.vstack([cness for cness in correctnesses])
inv_mask = np.logical_not(mask_matrix)

with open('cifar_mask.npy', 'wb') as f:
    np.save(f, mask_matrix)
with open('cifar_correctness.npy', 'wb') as f:
    np.save(f, correctness_matrix)

def masked_avg(x, mask, axis=0, esp=1e-10):
    return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

mem_est = masked_avg(correctness_matrix, mask_matrix) - masked_avg(correctness_matrix, inv_mask)
with open('cifar100_mem_est.npy', 'wb') as f:
    np.save(f, mem_est)




