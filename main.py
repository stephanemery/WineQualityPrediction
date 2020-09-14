import sys
import pandas as pd
from model.Preprocessing import preprocess
from model.MultiLinearRegression import MultiLinearRegression
from model.KNN import KNN
from model.SVM import SVM
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Parameters
    normalize = True
    remove_outliers = True
    scalerType='MinMaxScaler'
    test_size = 0.3

    # Preprocess
    preprocess(normalize, remove_outliers, scalerType)

    # Load data
    options = '_'
    if remove_outliers :
        options+='ro_'
    if normalize :
        options+='n_'

    red_wine = pd.read_csv('./data/preprocessed'+options+'red.csv')
    white_wine = pd.read_csv('./data/preprocessed'+options+'white.csv')
    
    # Split into train/test dataset
    train_set, test_set = train_test_split(red_wine, test_size=test_size, shuffle=True)
    
    # Models
    models = []
    # Add multi linear regression
    models.append(MultiLinearRegression())
    # Add KNN regressor
    models.append(KNN())
    # Add SVM regressor
    models.append(SVM(0, 0.5))

    # Train
    for m in models:
        m.train(train_set.to_numpy()[:, :-1], train_set.to_numpy()[:, -1])

    # Test 
    for m in models:
        m.test(test_set.to_numpy()[:, :-1], test_set.to_numpy()[:, -1])

    # Print scores
    for m in models:
        print(m)    
