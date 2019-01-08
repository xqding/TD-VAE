import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from model import *

checkpoint = torch.load("./output/model/model_epoch_199.pt")

input_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

with open("./data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)
tdvae.eval()

data = MNIST_Dataset(MNIST['train_image'])
image = data[10]
image = image.reshape([1, image.shape[0], image.shape[1]])
image = torch.tensor(image)

tdvae.forward(image)
idx = 5
z_mu = tdvae.state_mean[:,idx,:]
eps = z_mu.new_tensor(torch.randn(1,tdvae.state_size))
z = z_mu + tdvae.state_logstd[:,idx,:] * eps

x_p = tdvae.state_to_obs(z)
x_p = x_p.data.numpy().reshape(28,28)

fig = plt.figure(0)
plt.imshow(1-x_p, cmap = 'binary')

z_mu = tdvae.state_to_state_mean(z)
z_logstd = tdvae.state_to_state_logstd(z)
z = z_mu.new_tensor(torch.randn(1,tdvae.state_size))
x_p = tdvae.state_to_obs(z)
x_p = x_p.data.numpy().reshape(28,28)

fig = plt.figure(1)
plt.imshow(1-x_p, cmap = 'binary')

plt.show()
