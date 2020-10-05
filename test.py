import numpy as np
import pandas as pd
import nose.tools as nt
import os
import sys
import main

from os import path
from contextlib import contextmanager
from io import StringIO
from model.MultiLinearRegression import *
from model.Preprocessing import *
from model.KNN import *
from model.SVM import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def test_main_1():
    '''
    Test main function
    Test download data
    '''    
    # Remove file
    if path.exists("./data/winequality-red.csv"):
        os.remove("./data/winequality-red.csv")
    if path.exists("./data/winequality-white.csv"):
        os.remove("./data/winequality-white.csv")
    # Run main
    main.main(None, False)
    
    assert path.exists("./data/winequality-red.csv")
    assert path.exists("./data/winequality-white.csv")
        

def test_main_2():
    '''
    Test main file
    '''
    with captured_output() as (out, err):
        main.main(None, False)
        
    output = out.getvalue().strip()

    msg = 'Preprocessing done !\
\nMulti-Linear Regression\
\r\n\tScore : 0.252751\
\nKNN Regressor\
\r\n\tScore : 0.178368\
\nSVM Regressor\
\r\n\tScore : 0.247992'

    nt.assert_equal(output, msg)


def test_normalize_1():
    '''Test the normalize function'''
    x = np.random.rand(10,3)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    expected = (x - mean) / std
    result = normalize(pd.DataFrame(x), StandardScaler)
    
    np.testing.assert_array_almost_equal(
        expected, result.to_numpy(), decimal=6
    )
            
    
def test_normalize_2():
    '''Test the normalize function'''
    x = np.array([1])
            
    with nt.assert_raises(Exception):
        normalize(pd.DataFrame(x), StandardScaler)


def test_remove_outliers_1():
    '''Test remove outliers function'''
    x = pd.DataFrame(np.random.rand(100, 3))
    y = remove_outliers(x)
    assert (
        x.shape[0] * x.shape[1]
        > y.shape[0] * y.shape[1]
    )

def test_remove_outliers_2():    
    '''Test remove outliers function'''
    x = pd.DataFrame(np.arange(0, 100, 0.1))
    y = remove_outliers(x)
    
    assert y.to_numpy().ravel().all() == np.arange(1, 99, 0.1).all()

def test_features_selection_1():
    '''Test remove outliers function'''
    # Create a dataset with 5 features
    x = np.random.rand(100, 5)
    n_components = 3  
    # Keep only 3 features
    fs = features_selection(x, n_components)
    assert len(fs) == n_components
    
def test_preprocess_1():
    '''
    Test preprocess function
    Not normalize data and not remove outliers
    '''
    norm = False
    rm_outliers = False
    # Remove file
    if path.exists("./data/preprocessed_winequality-red.csv"):
        os.remove("./data/preprocessed_winequality-red.csv")
    # Preprocess data
    preprocess("./data/winequality-red.csv", norm, rm_outliers)
    
    assert path.exists("./data/preprocessed_winequality-red.csv")
    
def test_preprocess_2():
    '''
    Test preprocess function
    Normalize data
    '''
    norm = True
    rm_outliers = False
    # Remove file
    if path.exists("./data/preprocessed_n_winequality-red.csv"):
        os.remove("./data/preprocessed_n_winequality-red.csv")
    # Preprocess data
    preprocess("./data/winequality-red.csv", norm, rm_outliers)
    
    assert path.exists("./data/preprocessed_n_winequality-red.csv")
    
def test_preprocess_3():
    '''
    Test preprocess function
    Remove outliers
    '''
    norm = False
    rm_outliers = True
    # Remove file
    if path.exists("./data/preprocessed_ro_winequality-red.csv"):
        os.remove("./data/preprocessed_ro_winequality-red.csv")
    # Preprocess data
    preprocess("./data/winequality-red.csv", norm, rm_outliers)
    
    assert path.exists("./data/preprocessed_ro_winequality-red.csv")
    
def test_preprocess_4():
    '''
    Test preprocess function
    Normalize data and remove outliers
    '''
    norm = True
    rm_outliers = True
    # Remove file
    if path.exists("./data/preprocessed_ro_n_winequality-red.csv"):
        os.remove("./data/preprocessed_ro_n_winequality-red.csv")
    # Preprocess data
    preprocess("./data/winequality-red.csv", norm, rm_outliers)
    
    assert path.exists("./data/preprocessed_ro_n_winequality-red.csv")    
    
def test_KNN_1():
    '''
    Test KNN
    Test training and test
    '''
    np.random.seed(42)
    x = np.array([[7,4],[7,7], [7,8], [8,10], [11,6], [3,4],[1,4],[2,2], [1,3], [5,2], [3,3]])
    y = np.array([0,0,0,0,0,1,1,1,1,1,1])
    
    knn = KNN()
    knn.train(x, y)
    knn.test(x, y)
    
    assert knn.score > 0        
    assert knn.predict(x).all() == np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]).all()
    
def test_KNN_2():
    '''
    Test KNN
    Test test without training
    '''
    x = np.array([[7,4],[7,7], [7,8], [8,10], [11,6], [3,4],[1,4],[2,2], [1,3], [5,2], [3,3]])
    y = np.array([0,0,0,0,0,1,1,1,1,1,1])
    
    knn = KNN()
    knn.test(x, y)    

    assert knn.score == 0        
    assert knn.predict(x) is None

def test_MLR_1():
    '''
    Test MultiLinearRegression
    Test training and test
    '''
    np.random.seed(42)
    x = np.array([[7,4],[7,7], [7,8], [8,10], [11,6], [3,4],[1,4],[2,2], [1,3], [5,2], [3,3]])
    y = np.array([0,0,0,0,0,1,1,1,1,1,1])
    
    mlr = MultiLinearRegression()
    mlr.train(x, y)
    mlr.test(x, y)
    
    assert mlr.score > 0        
    np.testing.assert_array_almost_equal(
        mlr.predict(x), np.array([0.40948431, 0.17417959, 0.09574468, -0.16119726, -0.147674, 0.80977281,  1.00991706, 1.06671475, 1.08835197, 0.76649838, 0.88820772]), decimal=6)

def test_MLR_2():
    '''
    Test MultiLinearRegression
    Test test without training
    '''
    x = np.array([[7,4],[7,7], [7,8], [8,10], [11,6], [3,4],[1,4],[2,2], [1,3], [5,2], [3,3]])
    y = np.array([0,0,0,0,0,1,1,1,1,1,1])
    
    mlr = MultiLinearRegression()
    mlr.test(x, y)
    
    assert mlr.score == 0        
    assert mlr.predict(x) is None

def test_SVR_1():
    '''
    Test Support Vector Regression
    Test training and test
    '''
    np.random.seed(42)
    x = np.array([[7,4],[7,7], [7,8], [8,10], [11,6], [3,4],[1,4],[2,2], [1,3], [5,2], [3,3]])
    y = np.array([0,0,0,0,0,1,1,1,1,1,1])
    
    svr = SVM(0, 0.5)
    svr.train(x, y)
    svr.test(x, y)
    
    assert svr.score > 0        
    np.testing.assert_array_almost_equal(
        svr.predict(x), np.array([1.75000821e-01, 4.52573405e-04, -3.67081086e-04, 4.52569571e-04, -3.73049831e-05, 9.64221768e-01, 9.99864628e-01,  9.99511496e-01, 1.00012312e+00, 7.58637413e-01, 1.00580606e+00]), decimal=6)

def test_SVR_2():
    '''
    Test Support Vector Regression
    Test test without training
    '''
    x = np.array([[7,4],[7,7], [7,8], [8,10], [11,6], [3,4],[1,4],[2,2], [1,3], [5,2], [3,3]])
    y = np.array([0,0,0,0,0,1,1,1,1,1,1])
    
    svr = SVM(0, 0.5)
    svr.test(x, y)
    
    assert svr.score == 0        
    assert svr.predict(x) is None
    
@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


    