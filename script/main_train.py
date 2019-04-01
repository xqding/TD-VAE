__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 16:45:38"

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *
from prep_data import *
import sys

#### preparing dataset
with open("./data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST['train_image'])
batch_size = 512
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)

#### build a TD-VAE model
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
tdvae = tdvae.cuda()

#### training
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)
num_epoch = 4000
log_file_handle = open("./log/loginfo_new.txt", 'w')
for epoch in range(num_epoch):
    for idx, images in enumerate(data_loader):        
        images = images.cuda()       
        tdvae.forward(images)
        t_1 = np.random.choice(16)
        t_2 = t_1 + np.random.choice([1,2,3,4])
        loss = tdvae.calculate_loss(t_1, t_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
              file = log_file_handle, flush = True)
        
        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()))

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "./output/model/new_model_epoch_{}.pt".format(epoch))
