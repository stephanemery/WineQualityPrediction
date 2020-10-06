import sys
import pandas as pd
import requests
import os
import ntpath
import argparse

from .model.Preprocessing import preprocess
from .model.MultiLinearRegression import MultiLinearRegression
from .model.KNN import KNN
from .model.SVM import SVM
from sklearn.model_selection import train_test_split

def predictQuality(filepath=None, shuffle=True, normalize=True, remove_outliers=True, scalerType="StandardScaler", test_size=0.3, max_components=None):
    """
    Preprocess the data and try diffent model of learning on it.
    The function print the score of each model.
    """
    # If no filepath set, use the default one
    if filepath is None :
        filepath = 'wineQualityPred/data/winequality-red.csv'
        # If data files don't exist, download them
        # Red wine data
        if not os.path.isfile(filepath):
            print("Downloading red wine data...")
            my_file = requests.get(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            )
            f=open(filepath, "wb")
            f.write(my_file.content)
            f.close()
        path_white_wine = 'wineQualityPred/data/winequality-white.csv'
        # White wine data
        if not os.path.isfile(path_white_wine):
            print("Downloading white wine data...")
            my_file = requests.get(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
            )
            f=open(path_white_wine, "wb")
            f.write(my_file.content)
            f.close()

    # Preprocess
    preprocess(filepath, normalize, remove_outliers, scalerType, max_components)

    # Load data
    options = "_"
    if remove_outliers:
        options += "ro_"
    if normalize:
        options += "n_"

    data = pd.read_csv(ntpath.dirname(filepath)+"/preprocessed" + options + ntpath.basename(filepath))

    # Split into train/test dataset
    train_set, test_set = train_test_split(
        data, test_size=test_size, shuffle=shuffle
    )

    # Models
    models = []    
    # Add multi linear regression
    models.append(MultiLinearRegression())    
    # Add KNN regressor
    models.append(KNN())    
    # Add SVM regressor
    models.append(SVM(0, 0.5))    

    # Train models
    for m in models:
        m.train(train_set.iloc[:, :-1], train_set.iloc[:, -1])

    # Test models
    for m in models:
        m.test(test_set.iloc()[:, :-1], test_set.iloc()[:, -1])

    # Print scores
    for m in models:
        print(m)
    

def reproduceResults():
    predictQuality(None, False)
