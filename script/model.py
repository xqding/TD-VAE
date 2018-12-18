__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 18:05:21"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MNIST_Dataset(Dataset):
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        image = np.copy(self.image[idx, :].reshape(28,28))
        image = image.astype(np.float32)
        direction = np.random.choice(['left', 'right'])
        image_list = []
        image_list.append(image.reshape(-1))
        for k in range(1,20):
            if direction == 'left':
                tmp = np.delete(image, 0, 1)
                image = np.concatenate((tmp, image[:,0].reshape(-1,1)) ,1)
                image_list.append(image.reshape(-1))
            elif direction == 'right':
                tmp = np.delete(image, -1, 1)
                image = np.concatenate((image[:,-1].reshape(-1,1), tmp) ,1)
                image_list.append(image.reshape(-1))
                
        image_seq = np.array(image_list)
        return image_seq

class TD_VAE(nn.Module):
    def __init__(self, input_size, belief_state_size, state_size):
        super(TD_VAE, self).__init__()
        self.input_size = input_size
        self.belief_state_size = belief_state_size
        
        ## LSTM for aggregating belief states
        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.belief_state_size,
                            batch_first = True)

        ## belief state to state
        self.state_size = state_size
        self.belief_to_state_mean = nn.Sequential(
            nn.Linear(self.belief_state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.state_size))

        self.belief_to_state_logstd = nn.Sequential(
            nn.Linear(self.belief_state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.state_size))               
        
        ## state to observation
        self.state_to_obs = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.input_size),
            nn.Sigmoid())        

        ## state transition
        self.state_to_state_mean = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.state_size))
        
        self.state_to_state_logstd = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.state_size))

        ## inference state
        self.infer_state_mu = nn.Sequential(
            nn.Linear(self.belief_state_size*2 + self.state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.state_size))

        self.infer_state_logstd = nn.Sequential(
            nn.Linear(self.belief_state_size*2 + self.state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.state_size))        
        
    def forward(self, images):
        self.batch_size = images.shape[0]
        self.input = images
        
        self.belief_states, (h_n, c_n) = self.lstm(images)
        self.state_mean = self.belief_to_state_mean(self.belief_states)
        self.state_logstd = self.belief_to_state_logstd(self.belief_states)

    def calculate_loss(self, t_1, t_2):
        ## sample a state at time t_2
        self.epsilon_t_2 = self.state_mean.new_tensor(torch.randn(self.batch_size, self.state_size))
        self.state_t_2 = self.state_mean[:,t_2,:] + \
                         torch.exp(self.state_logstd[:,t_2,:])*self.epsilon_t_2

        ## inference a state at time t_1
        tmp = torch.cat((self.belief_states[:,t_1,:],
                         self.belief_states[:,t_2,:],
                         self.state_t_2), 1)
        self.state_t_1_mean = self.infer_state_mu(tmp)
        self.state_t_1_logstd = self.infer_state_logstd(tmp)
        
        self.epsilon_t_1 = self.state_mean.new_tensor(torch.randn(self.batch_size, self.state_size))
        self.state_t_1 = self.state_t_1_mean + \
                         torch.exp(self.state_t_1_logstd)*self.epsilon_t_1

        ####  calculate loss  ####
        ## p(x_2|z_2)
        obs_prop_t_2 = self.state_to_obs(self.state_t_2)
        self.nloss = torch.mean(torch.sum(
            self.input[:,t_2,:]*torch.log(obs_prop_t_2) + \
            (1-self.input[:,t_2,:])*torch.log(1-obs_prop_t_2), 1))

        ## -Pb(z_2|b_2)
        PI = self.belief_states.new_tensor(torch.tensor(np.pi))        
        self.nloss += -torch.mean(-torch.sum(self.epsilon_t_2**2, 1) - self.state_size/2*torch.log(2*PI) \
                                   -torch.sum(self.state_logstd[:,t_2,:], 1))

        ## Pb(z_1|b_1)
        self.nloss += torch.mean(torch.sum(-0.5*((self.state_t_1 - self.state_mean[:,t_1,:])/torch.exp(self.state_logstd[:,t_1,:]))**2, 1) - \
                   self.state_size/2*torch.log(2*PI) - torch.sum(self.state_logstd[:,t_1,:], 1))

        ## q(z_1|z_2,b_1,b_2)
        self.nloss += -torch.mean(-torch.sum(self.epsilon_t_1**2, 1) - self.state_size/2*torch.log(2*PI) \
                                   -torch.sum(self.state_t_1_logstd, 1))

        ## p(z_2|z_1)
        self.state_t_2_mean = self.state_to_state_mean(self.state_t_1)
        self.state_t_2_logstd = self.state_to_state_logstd(self.state_t_1)

        self.nloss += torch.mean(torch.sum(-0.5* ((self.state_t_2 - self.state_t_2_mean)/torch.exp(self.state_t_2_logstd))**2, 1) - \
                                 self.state_size/2*torch.log(2*PI) - torch.sum(self.state_t_2_logstd, 1))

        self.loss = -self.nloss

        return self.loss
