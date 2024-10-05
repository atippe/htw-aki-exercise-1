'''
Please load the winequality-white dataset and use a dense fully connected network to predict the quality of a wine.
Experiment with various options such as activation functions, learning rate, number of layers, number of hidden nodes per layer,
optimization algorithms, and loss functions. For each option plot the loss function over several epochs.
Submit your notebook and present your best results.
'''

import torch
import pandas

df = pandas.read_csv('winequality-white.csv', sep=';')

inputs = torch.tensor(df.iloc[:,0:11].values)
labels = torch.tensor(df.iloc[:,11:12].values)