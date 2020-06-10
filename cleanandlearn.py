import pandas as pd
import numpy as np 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import sys
import getopt

min_max_scaler = preprocessing.MinMaxScaler()


def keepBest(data):
    data = data[['dimension','budget', 'loss','optimizer_name']]
    new = data["optimizer_name"].str.split("_", n = 2, expand = True)
    data["mu"] = new[1] 
    data["lambda"] = new[2] 
    data["mu"] = data["mu"].astype('int64')
    data["lambda"] = data["lambda"].astype('int64')
    data.drop(columns =["optimizer_name"], inplace = True) 

    print(data.head())
    print(data.dtypes)
    alldata = np.unique(data[['dimension', 'budget', 'lambda']], axis=0)
    print(alldata.shape)
    cleaned_data = pd.DataFrame(columns=['dimension', 'budget', 'lambda', 'mu'])
    for a in alldata:
        d = a[0]
        b = a[1]
        l = a[2]
        tmp = data[data['dimension'] == d]
        tmp = tmp[tmp['budget'] == b]
        tmp = tmp[tmp['lambda'] == l]
        themin = tmp[tmp.loss == tmp.loss.min()]
        cleaned_data = cleaned_data.append(themin)
    del(cleaned_data['loss'])
    cleaned_data = cleaned_data.astype(dtype=float)
    x = cleaned_data.values #returns a numpy array
    x_scaled = min_max_scaler.fit(x)
    print(cleaned_data.shape)
    print(cleaned_data)
    return cleaned_data

def analyze(data):
    print(data)
    print(data.shape)
    print(data.info())
    print(data.describe())
    print(data.head())
    data.to_csv('prout.csv')
    # data.plot.scatter(x='mu', y='lambda');
    # plt.show()

    xx = 'mu'
    yy = 'lambda'
    zz = 'dimension'

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(data[xx], data[yy], data[zz])
    ax1.set_xlabel(xx)
    ax1.set_ylabel(yy)
    ax1.set_zlabel(zz)
    plt.show()


def splitData(data, test_ratio):
    train, test = train_test_split(data, test_size=test_ratio)
    x_train = train
    y_train = train['mu']
    x_scaled = min_max_scaler.transform(x_train)
    x_train = pd.DataFrame(x_scaled, columns=['dimension', 'budget', 'lambda', 'mu'])
    del(x_train['mu'])
    x_test = test
    y_test = test['mu']
    x_scaled = min_max_scaler.transform(x_test)
    x_test = pd.DataFrame(x_scaled, columns=['dimension', 'budget', 'lambda', 'mu'])
    del(x_test['mu'])
    return x_train, y_train, x_test, y_test


def create_model(classifier, x, y):
    classifier.fit(x, y)
    return classifier


def display_score(classifier, x_train, y_train, x_test, y_test):
    y_pred = classifier.predict(x_test)
    print('Coefficient of determination: %s' % r2_score(y_test, y_pred))
    print('MAE: %s' % mean_absolute_error(y_test, y_pred))
    print('MSE: %s' % mean_squared_error(y_test, y_pred))


###############################################
###### With pytorch
###############################################
class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.inputSize = 3
        self.outputSize = 1
        # self.hiddenSize1 = 30
        # self.hiddenSize2 = 80
        # self.hiddenSize3 = 20
        # self.hiddenSize4 = 5
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.inputSize, self.hiddenSize1),
        #     # nn.BatchNorm1d(self.hiddenSize1),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(self.hiddenSize1, self.hiddenSize2),
        #     # nn.BatchNorm1d(self.hiddenSize2),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(self.hiddenSize2, self.hiddenSize3),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(self.hiddenSize3, self.hiddenSize4),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(self.hiddenSize4, self.outputSize)
        # )
        
        self.hiddenSize1 = 10
        self.hiddenSize2 = 10
        self.classifier = nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize1),
            # nn.BatchNorm1d(self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize1, self.hiddenSize2),
            # nn.BatchNorm1d(self.hiddenSize1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hiddenSize2, self.outputSize)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

def usage():
    print("Usage: ./cleanandlearn --filename=data_file_name.csv")

def main(argv):
    filename=''
    try:                                
        opts, args = getopt.getopt(argv, "n", ["filename="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    if (len(opts) == 0):
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--filename'):
            filename = arg


    data = pd.read_csv(filename, sep=',')
    cleaned_data = keepBest(data)
    analyze(cleaned_data)
    return
    x_train, y_train, x_test, y_test = splitData(cleaned_data, 0.2)
    print(x_train.shape)
    print(x_test.shape)
    model = Model()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.RMSprop(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)


    data_train = torch.utils.data.TensorDataset(torch.Tensor(np.array(x_train)), torch.Tensor(np.array(y_train)))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size = 64, shuffle = True)

    data_test = torch.utils.data.TensorDataset(torch.Tensor(np.array(x_test)), torch.Tensor(np.array(y_test)))
    test_loader = torch.utils.data.DataLoader(data_test)

    for epoch in range(1, 10001): ## run the model for 10 epochs
        train_loss, valid_loss = [], []

        ## training part 
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()

            ## 1. forward propagation
            output = model(data)
            ## 2. loss calculation
            loss = loss_function(output, target.view(-1,1))
            ## 3. backward propagation
            loss.backward()
            ## 4. weight optimization
            optimizer.step()

            train_loss.append(loss.item())

        ## evaluation part 
        model.eval()
        for data, target in test_loader:
            output = model(data)
            loss = loss_function(output, target.view(-1,1))
            valid_loss.append(loss.item())

        print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
    model.eval()
    for data, target in test_loader:
        pred = model(data)
        pred2 = pred.detach().numpy()
        target2 = target.detach().numpy()
        # pred2 = min_max_scaler.inverse_transform(pred2)
        # target2 = min_max_scaler.inverse_transform(target2)
        print(pred2)
        print(target2)


if __name__ == '__main__':
    main(sys.argv[1:])
