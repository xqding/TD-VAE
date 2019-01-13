import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MNIST_Dataset(Dataset):
    def __init__(self, image, binary = True):
        super(MNIST_Dataset).__init__()
        self.image = image
        self.binary = binary
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        image = np.copy(self.image[idx, :].reshape(28,28))
        
        if self.binary:
            # ## binarize MNIST images
            tmp = np.random.rand(28,28)
            image = tmp <= image
            
        image = image.astype(np.float32)

        ## randomly choose a direction and generate a sequence
        ## of images that move in the chosen direction
        direction = np.random.choice(['left', 'right'])
        image_list = []
        #image = np.roll(image, np.random.choice(np.arange(28)), 1)
        image_list.append(image.reshape(-1))
        for k in range(1,20):
            if direction == 'left':
                image = np.roll(image, -1, 1)
                image_list.append(image.reshape(-1))
            elif direction == 'right':
                image = np.roll(image, 1, 1)
                image_list.append(image.reshape(-1))
                
        image_seq = np.array(image_list)
        return image_seq
