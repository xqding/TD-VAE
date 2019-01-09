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
        
        ## binarize MNIST images
        tmp = np.random.rand(28,28)
        image = tmp <= image
        image = image.astype(np.float32)

        ## randomly choose a direction and generate a sequence
        ## of images that move in the chosen direction
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
