# Temporal Difference Variational Auto-Encoder (TD-VAE) (Implemented using PyTorch)

This is an implementation of the TD-VAE introduced in [this ICLR 2019 paper](https://openreview.net/forum?id=S1x4ghC9tQ), which is really well writen. 
TD-VAE is designed such that it have all following three features:

1. It learns a state representation of observations and makes predictions on the state level.
2. Based on observations, it learns a belief state that contains all the inforamtion required to make predictions about the future.
3. It learns to make predictions multiple steps in the future directly instead of make predictions step by step. It learns by connecting states that are multiple steps apart.

Here, based on the information disclosed in the paper, I try to reproduce the experiment about moving MNIST digits. In this experiment, a sequence of a MNIST digit moving to the left or the right direction is presented to the model. The model need to predict how the digit moves in the following steps. After training the model, we can feed a sequence of digit into the model and see how well it can predict the further. Here is the result.
![Figure](./output/rollout_result.png)
