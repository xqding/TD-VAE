__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 22:29:44"

from prep_data import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec

with open("./data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

#np.random.seed(1)
data = MNIST_Dataset(MNIST['train_image'])
fig = plt.figure(0, figsize = (20,3))
fig.clf()
gs = gridspec.GridSpec(4,20)
gs.update(wspace = 0.025, hspace = 0.025)

for i in range(4):
    a = data[i]
    for n in range(20):
        axes = plt.subplot(gs[i,n])
        axes.imshow(1-a[n,:].reshape(28,28), cmap = 'binary', vmax = 1, vmin = 0)
        axes.axis('off')
plt.show()
