# regression using torch library(ann)
# Mehmet VARAN
# -*- coding: utf-8 -*-

# importing libraries
import torch
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# setting seed value
torch.manual_seed(5)

# getting raw data and splitting as train and test
data = pd.read_csv('robot_ds.csv', sep = ';')
data_train, data_test = train_test_split(data, test_size= 0.25, random_state=5)

data_train.to_csv('robot_train.csv', index=False)
data_test.to_csv('robot_test.csv', index=False)


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path)
        # store the inputs and outputs
        self.X = df.values[:, 11:-4].astype('float32')
        self.y = df.values[:, -4:-1].astype('float32')
        # printing datas separetly
        self.y = self.y.reshape((len(self.y), 3))
        print(self.X)
        print(self.y)
        
        
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.25):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
    
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 100)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(100, 30)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        self.hidden3 = Linear(30, 3)
        xavier_uniform_(self.hidden3.weight)
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X

def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl): # 
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            
            # evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 3))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse

# making prediction for given data
def predict(Dataset, model):
    # convert row to data
    data_pred = torch.tensor(Dataset.values)
    data_pred= data_pred.type(torch.FloatTensor)
    # make prediction
    yhat = model(data_pred)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# getting train data 
path = 'robot_train.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(200)

# train the model
train_model(train_dl, model)

# evaluate the model
mse = evaluate_model(test_dl, model)
print('MSE: %.4f, RMSE: %.4f' % (mse, sqrt(mse)))

# getting test(never used) data, separating into inputs and XYZ
data_test = read_csv('robot_test.csv')
data_testFinal = data_test.iloc[:,10:-1]
data_testInputs = data_testFinal.iloc[:,1:-3]
data_testXYZ = data_test.iloc[:,-4:-1]

# making predictions and printing r2 score
yhat = predict(data_testInputs, model)
print('Predicted:', yhat)
print("R2 Score: %.4f" % r2_score(data_testXYZ, yhat))

# yhat is predicted data / data_testXYZ is real data

