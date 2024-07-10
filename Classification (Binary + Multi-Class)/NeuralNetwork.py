import numpy as np;
import pandas as pd;
import sklearn;
import matplotlib.pyplot as plotter;
import os;

from sklearn import preprocessing;
from sklearn import utils;
from sklearn import metrics;

import tensorflow as tf;

os.system('cls');

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

def LossPlot(History):

    Figure, Axis = plotter.subplots(1, 1);

    Axis.plot(History.history['loss'], label = 'Training Loss');

    Axis.plot(History.history['val_loss'], label = 'Validation Loss');

    Axis.set_xlabel('Epoch');

    Axis.set_ylabel('Binary Cross Entropy Loss');

    Axis.grid(True);

    plotter.legend();

    plotter.show();

def AccuracyPlot(History):

    Figure, Axis = plotter.subplots(1, 1);

    Axis.plot(History.history['accuracy'], label = 'Training Accuracy');

    Axis.plot(History.history['val_accuracy'], label = 'Validation Accuracy');

    Axis.set_xlabel('Epoch');

    Axis.set_ylabel('Accuracy');

    Axis.grid(True);

    plotter.legend();

    plotter.show();

FileName = input("Enter the name of the DataSet: ");

print();

Categorials = input("Enter the name of All Categorial Variables & The Target Variable: ").split();

print();

Dataset = pd.read_csv(FileName);

Dataset = pd.get_dummies(Dataset, columns= Categorials);

Train, Valid, Test = np.split(Dataset.sample(frac=1), [int(.6*len(Dataset)), int(.8*len(Dataset))]);

os.system('cls');

print("The Dataset: -\n");

print(Dataset.head(), '\n');

Train, TrainF, TrainO = ScaleX(Train, True);

Valid, ValidF, ValidO = ScaleX(Valid);

Test, TestF, TestO = ScaleX(Test);  

print("Creating a Multi-layer NN Model: -\n")

neurons = int(input("Enter No. of Nodes for Each Layer: "));

print();

dropout_probability = float(input("Enter the Dropout Probability: " ));

print();

Epochs = int(input("Enter the Number of Epochs: "));

print();

NNModel = tf.keras.models.Sequential([
    tf.keras.layers.Dense(neurons, activation = 'relu'),
    tf.keras.layers.Dropout(dropout_probability),
    tf.keras.layers.Dense(neurons, activation = 'relu'),
    tf.keras.layers.Dropout(dropout_probability),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
]);

NNModel.compile(optimizer = tf.keras.optimizers.Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy']);

History = NNModel.fit(TrainF, TrainO, epochs = Epochs, validation_data = (ValidF, ValidO));

LossPlot(History);

AccuracyPlot(History);

Predictions = NNModel.predict(TestF);

Predictions = np.round(Predictions);

Report = sklearn.metrics.classification_report(TestO, Predictions);

print("\nClassification Report using a Neural Network Model: -\n");

print(Report);

Accuracy = sklearn.metrics.accuracy_score(TestO, Predictions);

print("Specifications For The Network: \n");

print(NNModel.summary(), '\n');

print("\nAccuracy Score For This Neural Network Model is: ", Accuracy * 100, "%", '\n');