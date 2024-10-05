'''
Please load the winequality-white dataset and use a dense fully connected network to predict the quality of a wine.
Experiment with various options such as activation functions, learning rate, number of layers, number of hidden nodes per layer,
optimization algorithms, and loss functions. For each option plot the loss function over several epochs.
Submit your notebook and present your best results.
'''

import matplotlib.pyplot as plt
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pandas.read_csv('winequality-white.csv', sep=';')

df_train, df_test = train_test_split(df, test_size=0.2)

inputs_train = torch.tensor(df_train.iloc[:,0:11].values)
labels_train = torch.tensor(df_train.iloc[:,11:12].values)

inputs_test = torch.tensor(df_test.iloc[:,0:11].values)
labels_test = torch.tensor(df_test.iloc[:,11:12].values)

# scale the inputs
min_max_scaler = preprocessing.MinMaxScaler()

# for training
min_max_scaler.fit(inputs_train) # fit ONCE on the training data only
inputs_train_scaled = min_max_scaler.transform(inputs_train)
inputs_train_scaled = torch.tensor(inputs_train_scaled)

# for testing transform using the SAME scaler
inputs_test_scaled = min_max_scaler.transform(inputs_test) # only transform, no fitting
inputs_test_scaled = torch.tensor(inputs_test_scaled)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(11, 96)
        self.layer_2 = nn.Linear(96, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x

net = Net()

print(net.layer_1.weight)
print(net.layer_2.weight)
print(net.output_layer.weight)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_values_train = []
loss_values_test = []

for epoch in range(300):
    optimizer.zero_grad()
    train_outputs = net(inputs_train_scaled.float())
    train_loss = criterion(train_outputs, labels_train.float())
    train_loss.backward()
    optimizer.step()
    loss_values_train.append(train_loss.item())

    with torch.no_grad():
        net.eval()
        test_outputs = net(inputs_test_scaled.float())
        test_loss = criterion(test_outputs, labels_test.float())
        loss_values_test.append(test_loss.item())
    net.train()

plt.plot(range(300), loss_values_train,
         label=f'train (final: {loss_values_train[-1]:.3f})',
         color='blue')
plt.plot(range(300), loss_values_test,
         label=f'test (final: {loss_values_test[-1]:.3f})',
         color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and test loss over epochs')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()


print(f"Loss Value Training: {loss_values_train[-1]}")
print(f"Loss Value Test: {loss_values_test[-1]}")