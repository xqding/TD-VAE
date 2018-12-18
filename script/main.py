__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 16:45:38"

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *

with open("./data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST['train_image'])
data_loader = DataLoader(data, batch_size = 300,
                         shuffle = True)

input_size = 784
belief_state_size = 50
state_size = 50 
tdvae = TD_VAE(input_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters())

for epoch in range(20):
    for idx, images in enumerate(data_loader):
        t_1 = np.random.choice(16)
        t_2 = t_1 + np.random.choice([1,2,3,4])
        tdvae.forward(images)
        loss = tdvae.calculate_loss(t_1, t_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()))
        
        
    

