__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 18:05:21"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

class PreProcess(nn.Module):
    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t

class Decoder(nn.Module):
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)
        
    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p

    
class TD_VAE(nn.Module):
    def __init__(self, x_size, processed_x_size, b_size, z_size):
        super(TD_VAE, self).__init__()
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size

        ## input pre-process layer
        self.process_x = PreProcess(self.x_size, self.processed_x_size)
        
        ## LSTM for aggregating belief states
        self.lstm = nn.LSTM(input_size = self.processed_x_size,
                            hidden_size = self.b_size,
                            batch_first = True)
        
        ## belief to state
        self.l2_b_to_z = DBlock(b_size, 50, z_size)
        self.l1_b_to_z = DBlock(b_size + z_size, 50, z_size)
        
        ## infer state
        self.l2_infer_z = DBlock(b_size + 2*z_size, 50, z_size)
        self.l1_infer_z = DBlock(b_size + 2*z_size + z_size, 50, z_size)

        ## state transition
        self.l2_transition_z = DBlock(2*z_size, 50, z_size)
        self.l1_transition_z = DBlock(2*z_size + z_size, 50, z_size)

        ## state to observation
        self.z_to_x = Decoder(2*z_size, 200, x_size)


    def forward(self, images):
        self.batch_size = images.size()[0]
        self.x = images
        self.processed_x = self.process_x(self.x)
        self.b, (h_n, c_n) = self.lstm(self.processed_x)
        
    def calculate_loss(self, t1, t2):
        ## sample a state at time t2
        t2_l2_z_mu, t2_l2_z_logsigma = self.l2_b_to_z(self.b[:, t2, :])        
        t2_l2_z_epsilon = torch.randn_like(t2_l2_z_mu)
        t2_l2_z = t2_l2_z_mu + torch.exp(t2_l2_z_logsigma)*t2_l2_z_epsilon
        
        t2_l1_z_mu, t2_l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t2,:], t2_l2_z),dim = -1))
        t2_l1_z_epsilon = torch.randn_like(t2_l1_z_mu)
        t2_l1_z = t2_l1_z_mu + torch.exp(t2_l1_z_logsigma)*t2_l1_z_epsilon
        
        t2_z = torch.cat((t2_l1_z, t2_l2_z), dim = -1)

        ## infer state at time t1 based on states at time t2
        t1_l2_qs_z_mu, t1_l2_qs_z_logsigma = self.l2_infer_z(
            torch.cat((self.b[:,t1,:], t2_z), dim = -1))
        t1_l2_qs_z_epsilon = torch.randn_like(t1_l2_qs_z_mu)
        t1_l2_qs_z = t1_l2_qs_z_mu + torch.exp(t1_l2_qs_z_logsigma)*t1_l2_qs_z_epsilon

        t1_l1_qs_z_mu, t1_l1_qs_z_logsigma = self.l1_infer_z(
            torch.cat((self.b[:,t1,:], t2_z, t1_l2_qs_z), dim = -1))
        t1_l1_qs_z_epsilon = torch.randn_like(t1_l1_qs_z_mu)
        t1_l1_qs_z = t1_l1_qs_z_mu + torch.exp(t1_l1_qs_z_logsigma)*t1_l1_qs_z_epsilon

        t1_qs_z = torch.cat((t1_l1_qs_z, t1_l2_qs_z), dim = -1)

        ## state at time t1 based on belief at time 1
        t1_l2_pb_z_mu, t1_l2_pb_z_logsigma = self.l2_b_to_z(self.b[:, t1, :])        
        t1_l1_pb_z_mu, t1_l1_pb_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t1,:], t1_l2_qs_z),dim = -1))

        ## state at time t2 based on states at time t1
        t2_l2_t_z_mu, t2_l2_t_z_logsigma = self.l2_transition_z(t1_qs_z)
        t2_l1_t_z_mu, t2_l1_t_z_logsigma = self.l1_transition_z(
            torch.cat((t1_qs_z, t2_l2_z), dim = -1))
        
        ## observation at time t2 based on state at time t2
        t2_x_prob = self.z_to_x(t2_z)


        #### calculate loss  ####
        ## KL divergence between t1_l2_pb_z, and t1_l2_qs_z
        loss = 0.5*torch.sum(((t1_l2_pb_z_mu - t1_l2_qs_z)/torch.exp(t1_l2_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l2_pb_z_logsigma, -1) - torch.sum(t1_l2_qs_z_logsigma, -1)

        ## KL divergence between t1_l1_pb_z and t1_l1_qs_z
        loss += 0.5*torch.sum(((t1_l1_pb_z_mu - t1_l1_qs_z)/torch.exp(t1_l1_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l1_pb_z_logsigma, -1) - torch.sum(t1_l1_qs_z_logsigma, -1)

        ## state log probabilty at time t2 based on belief
        loss += torch.sum(-0.5*t2_l2_z_epsilon**2 - 0.5*t2_l2_z_epsilon.new_tensor(2*np.pi) - t2_l2_z_logsigma, dim = -1) 
        loss += torch.sum(-0.5*t2_l1_z_epsilon**2 - 0.5*t2_l1_z_epsilon.new_tensor(2*np.pi) - t2_l1_z_logsigma, dim = -1)

        ## state log probabilty at time t2 based on transition
        loss += torch.sum(0.5*((t2_l2_z - t2_l2_t_z_mu)/torch.exp(t2_l2_t_z_logsigma))**2 + 0.5*t2_l2_z.new_tensor(2*np.pi) + t2_l2_t_z_logsigma, -1)
        loss += torch.sum(0.5*((t2_l1_z - t2_l1_t_z_mu)/torch.exp(t2_l1_t_z_logsigma))**2 + 0.5*t2_l1_z.new_tensor(2*np.pi) + t2_l1_t_z_logsigma, -1)

        ## observation prob at time t2
        loss += -torch.sum(self.x[:,t2,:]*torch.log(t2_x_prob) + (1-self.x[:,t2,:])*torch.log(1-t2_x_prob), -1)
        loss = torch.mean(loss)
        
        return loss
