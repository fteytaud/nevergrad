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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
import keras
min_max_scaler = preprocessing.MinMaxScaler()


def keepBest(data, k=0):
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
        tmp2 = tmp['loss']
        print(tmp2.shape)
        tmp2 = tmp2.sort_values(ascending=True)
        tmp2 = tmp2.reset_index(drop=True)
        v_min = tmp2[k]
        themin = tmp[tmp.loss <= v_min]
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
    # data.plot.scatter(x='mu', y='lambda');
    # plt.show()

    xx = 'mu'
    yy = 'lambda'
    zz = 'dimension'

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.scatter(data[xx], data[yy], data[zz])
    # ax1.set_xlabel(xx)
    # ax1.set_ylabel(yy)
    # ax1.set_zlabel(zz)
    # plt.show()

    # dimension = 2
    # datatmp = data[data["dimension"]==dimension]
    # for budget in [30, 50, 100, 300, 600, 900, 1500, 2000, 2500, 3000]:
    #     data2 = datatmp[datatmp["budget"]==budget]
    #     plt.plot(data2[yy], np.log(data2[xx]), label='budget={}'.format(budget))
    #     print("Budget {} : {} : {}".format(budget, data2[yy], data2[xx]))
    # plt.legend()
    # plt.show()

    # dimension = 2
    # datatmp = data[data["dimension"]==dimension]
    # for budget in [30, 50, 100, 300, 600, 900, 1500, 2000, 2500, 3000]:
    #     data2 = datatmp[datatmp["budget"]==budget]
    #     plt.boxplot(data2[yy], np.log(data2[xx]))
    #     print("Budget {} : {} : {}".format(budget, data2[yy], data2[xx]))
    # legend = ['budget={}'.format(budget) for budget in [30, 50, 100, 300, 600, 900, 1500, 2000, 2500, 3000]]
    # plt.legend(legend)
    # plt.show()

    dimension = 2
    datatmp = data[data["dimension"]==dimension]
    lambdas = np.unique(data[['lambda']], axis=0)
    data2 = datatmp[datatmp["budget"]==30]
    data2 = data2.drop(['dimension', 'budget'], axis=1)
    lambdas = lambdas.reshape(-1)
    for l in lambdas:
        data3 = data2[data2['lambda']==l].copy()
        data3 = data3['mu']#.drop(['lambda'], axis=1)
        plt.boxplot(data3, positions=[l])
    plt.show()

    data.to_csv('prout.csv')
    data2.to_csv('prout2.csv')


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



def batch_classify(_dict_classifiers, _X_train, _Y_train, _X_test, _Y_test, verbose=True):
    # Apprentissage sur un dictionnaire de classifieurs
    df_results = pd.DataFrame(data=np.zeros(shape=(len(_dict_classifiers.keys()), 4)),
                              columns=['classifier', 'train_score', 'test_score', 'training_time'])
    count = 0
    for key, classifier in _dict_classifiers.items():
        time_taken = time.perf_counter()
        classifier.fit(_X_train, _Y_train)
        time_taken = time.perf_counter() - time_taken
        train_score = classifier.score(_X_train, _Y_train)
        test_score = classifier.score(_X_test, _Y_test)
        df_results.loc[count, 'classifier'] = key
        df_results.loc[count, 'train_score'] = train_score
        df_results.loc[count, 'test_score'] = test_score
        df_results.loc[count, 'training_time'] = time_taken
        if verbose:
            print("Trained {c} in {f:.2f} s".format(c=key, f=time_taken))
        count += 1

    return df_results

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
    cleaned_data = keepBest(data, 50)
    analyze(cleaned_data)
    return

    x_train, y_train, x_test, y_test = splitData(cleaned_data, 0.2)
    print(x_train.shape)
    print(x_test.shape)
    # Dictionnaire des classifieurs que l'on souhaite tester
    dict_classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Nearest Neighbors": KNeighborsRegressor(),
        "Linear SVM": SVC(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Decision Tree": tree.DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=10000, max_depth=6),
        "Neural Net": MLPRegressor(alpha=1),
        "Naive Bayes": GaussianNB()
    }

    results = batch_classify(dict_classifiers, x_train, y_train, x_test, y_test)
    print(results.sort_values(by='test_score', ascending=False))



if __name__ == '__main__':
    main(sys.argv[1:])
