import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from model import *
from prep_data import *

checkpoint = torch.load("./output/model/model_epoch_3799.pt")

input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

with open("./data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)
tdvae.eval()
tdvae = tdvae.cuda()

data = MNIST_Dataset(MNIST['train_image'])
batch_size = 5
data_loader = DataLoader(data, batch_size = batch_size,
                         shuffle = True)

for idx, images in enumerate(data_loader):
    images = images.cuda()                       
    tdvae.forward(images)

    t1 = 12
    
    t1_l2_z_mu, t1_l2_z_logsigma = tdvae.l2_b_to_z(tdvae.b[:,t1,:])
    t1_l2_z_epsilon = torch.randn_like(t1_l2_z_mu)
    t1_l2_z = t1_l2_z_mu + torch.exp(t1_l2_z_logsigma)*t1_l2_z_epsilon

    t1_l1_z_mu, t1_l1_z_logsigma = tdvae.l1_b_to_z(
        torch.cat((tdvae.b[:,t1,:], t1_l2_z), dim = -1))
    t1_l1_z_epsilon = torch.randn_like(t1_l1_z_mu)
    t1_l1_z = t1_l1_z_mu + torch.exp(t1_l1_z_logsigma)*t1_l1_z_epsilon

    x_list = []
    t1_z = torch.cat((t1_l1_z, t1_l2_z), dim = -1)
    t1_x = tdvae.z_to_x(t1_z)    
    x_list.append(t1_x)
    
    for k in range(4):
        t2_l2_z_mu, t2_l2_z_logsigma = tdvae.l2_transition_z(t1_z)
        t2_l2_z_epsilon = torch.randn_like(t2_l2_z_mu)
        t2_l2_z = t2_l2_z_mu + torch.exp(t2_l2_z_logsigma)*t2_l2_z_epsilon

        t2_l1_z_mu, t2_l1_z_logsigma  = tdvae.l1_transition_z(
            torch.cat((t1_z, t2_l2_z), dim = -1))
        t2_l1_z_epsilon = torch.randn_like(t2_l1_z_mu)        
        t2_l1_z = t2_l1_z_mu + torch.exp(t2_l1_z_logsigma)*t2_l1_z_epsilon

        t2_z = torch.cat((t2_l1_z, t2_l2_z), dim = -1)
        
        t2_x = tdvae.z_to_x(t2_z)
        x_list.append(t2_x)
        
        t1_z = t2_z

    fig = plt.figure(0, figsize = (t1+5,batch_size))
    fig.clf()
    gs = gridspec.GridSpec(batch_size,t1+5)
    gs.update(wspace = 0.025, hspace = 0.025)
    for i in range(batch_size):        
        for j in range(t1):
            axes = plt.subplot(gs[i,j])
            axes.imshow(1-images.cpu().data.numpy()[i,j].reshape(28,28),
                        cmap = 'binary')
            axes.axis('off')

        for j in range(5):
            axes = plt.subplot(gs[i,t1+j])
            axes.imshow(1-x_list[j].cpu().data.numpy()[i,:].reshape(28,28),
                        cmap = 'binary')
            axes.axis('off')
            
    plt.show()
    break
    
    
    
# sys.exit()
# image = data[10]
# image = image.reshape([1, image.shape[0], image.shape[1]])
# image = torch.tensor(image)

# tdvae.forward(image)
# idx = 5
# z_mu = tdvae.state_mean[:,idx,:]
# eps = z_mu.new_tensor(torch.randn(1,tdvae.state_size))
# z = z_mu + tdvae.state_logstd[:,idx,:] * eps

# x_p = tdvae.state_to_obs(z)
# x_p = x_p.data.numpy().reshape(28,28)

# fig = plt.figure(0)
# plt.imshow(1-x_p, cmap = 'binary')

# z_mu = tdvae.state_to_state_mean(z)
# z_logstd = tdvae.state_to_state_logstd(z)
# z = z_mu.new_tensor(torch.randn(1,tdvae.state_size))
# x_p = tdvae.state_to_obs(z)
# x_p = x_p.data.numpy().reshape(28,28)

# fig = plt.figure(1)
# plt.imshow(1-x_p, cmap = 'binary')

# plt.show()
