import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from cnn import CNN
from utils import subset_train

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)

num_splits = 500

for i in range(num_splits):
    perm = torch.randperm(len(fashion)).numpy()
    
    split1 = perm[:(len(fashion) // 2)]
    split2 = perm[(len(fashion) // 2):]

    with open(f'./subsets/split_{2*i}', 'wb') as f:
        np.save(f, split1)
    with open(f'./subsets/split_{2*i + 1}', 'wb') as f:
        np.save(f, split2)
    
    net1 = subset_train(fashion, split1, device=device)
    net2 = subset_train(fashion, split2, device=device)

    torch.save(net1.state_dict(), f'./models/split_{2*i}')
    torch.save(net2.state_dict(), f'./models/split_{2*i + 1}')
