import numpy as np;
import pandas as pd;
import sklearn;
import os;

from sklearn import neighbors;
from sklearn import naive_bayes;
from sklearn import linear_model;
from sklearn import svm;

FileName = input("Enter the name of the DataSet: ");

Target = input("Enter the name of the Target Variable: ");

Dataset = pd.read_csv(FileName);

Train, Valid, Test = np.split(Dataset.sample(frac=1), [int(.6*len(Dataset)), int(.8*len(Dataset))]);

os.system('cls');

print("The Dataset: -\n");

print(Dataset.head(), '\n');

def ScaleX(dataframe, Oversample = False):
    
    Features = dataframe[dataframe.columns[:-1]].values;
    Output = dataframe[dataframe.columns[-1]].values;

    Scaler = sklearn.preprocessing.StandardScaler();
    Features = Scaler.fit_transform(Features);

    if Oversample:
        Oversampler = sklearn.utils.resample;
        Features, Output = Oversampler(Features, Output);

    Data = np.hstack((Features, np.reshape(Output, (-1, 1))));

    return Data, Features, Output;

def KNN_Model(TrainF, TrainO, TestF, TestO):

    X = int(input("Enter the number of neighbours: "));

    KNN = neighbors.KNeighborsClassifier(n_neighbors = X);

    KNN.fit(TrainF, TrainO);

    Predictions = KNN.predict(TestF);

    Report = sklearn.metrics.classification_report(TestO, Predictions);

    print("\nClassification Report using a KNN Model with", X, "Neighbours: -\n");

    print(Report);

    Accuracy = sklearn.metrics.accuracy_score(TestO, Predictions);

    print("\nAccuracy Score For This KNN Model is: ", Accuracy * 100, "%", '\n');

    return;

def NB_Model(TrainF, TrainO, TestF, TestO):

    Bayes = naive_bayes.GaussianNB();

    Bayes.fit(TrainF, TrainO);

    Predictions = Bayes.predict(TestF);

    Report = sklearn.metrics.classification_report(TestO, Predictions);

    print("\nClassification Report using a Naive Bayes Model: -\n");

    print(Report);

    Accuracy = sklearn.metrics.accuracy_score(TestO, Predictions);

    print("\nAccuracy Score For This Naive Bayes Model is: ", Accuracy * 100, "%", '\n');

    return;

def Logistic_Model(TrainF, TrainO, TestF, TestO):

    Logistic = linear_model.LogisticRegression();

    Logistic.fit(TrainF, TrainO);

    Predictions = Logistic.predict(TestF);

    Report = sklearn.metrics.classification_report(TestO, Predictions);

    print("\nClassification Report using a Logistic Regression Model: -\n");

    print(Report);

    Accuracy = sklearn.metrics.accuracy_score(TestO, Predictions);

    print("\nAccuracy Score For This Logistic Regression Model is: ", Accuracy * 100, "%", '\n');

    return;

def SupportVectorMachine_Model(TrainF, TrainO, TestF, TestO):

    SupportVectorMachines = svm.SVC();

    SupportVectorMachines.fit(TrainF, TrainO);

    Predictions = SupportVectorMachines.predict(TestF);

    Report = sklearn.metrics.classification_report(TestO, Predictions);

    print("\nClassification Report using a Support Vector Machine Model: -\n");

    print(Report);

    Accuracy = sklearn.metrics.accuracy_score(TestO, Predictions);

    print("\nAccuracy Score For This Support Vector Machine Model is: ", Accuracy * 100, "%", '\n');

    return;

print(" Dataset Sizes Before Oversampling and/or Scaling: -\n");

print("Training : Validation : Testing => ", len(Train), len(Valid), len(Test), '\n');

Train, TrainF, TrainO = ScaleX(Train, True);

Valid, ValidF, ValidO = ScaleX(Valid);

Test, TestF, TestO = ScaleX(Test);

print("Oversampling & Scaling Completed...\n");

KNN_Model(TrainF, TrainO, TestF, TestO);

NB_Model(TrainF, TrainO, TestF, TestO);

Logistic_Model(TrainF, TrainO, TestF, TestO);

SupportVectorMachine_Model(TrainF, TrainO, TestF, TestO);