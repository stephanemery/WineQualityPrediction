import numpy as np
import pandas as pd

from model.MultiLinearRegression import *
from model.Preprocessing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from hypothesis import given, strategies as st

Values = st.integers()
list1 = st.lists(Values)

@given(x=list1)
def test_normalize_1(x):
    x = np.array(x)
    if len(x)>1 :
        mean = np.mean(x, axis=0)
        std = np.std  (x, axis=0)
        if(std!=0):
            expected =    (x-mean)/std
            result = normalize(pd.DataFrame(x), StandardScaler)   
            np.testing.assert_array_almost_equal(expected, result.to_numpy().ravel(), decimal=6)

Values = st.integers()
list1 = st.lists(Values)

@given(x=list1)
def test_remove_outliers_1(x):
    x = np.array(x)
    x.copy()
    assert(x.shape[0] >= remove_outliers(x).shape[0])  

Values = st.integers()
list1 = st.lists(Values)
#TODO create 2D dataset
@given(x=list1)
def test_remove_outliers_2(x):
    x = np.array(x)
    x.copy()
    assert(x.shape[0] >= remove_outliers(x).shape[0])  

'''
# Parameters
normalize = True
remove_outliers = True
scalerType='StandardScaler'
test_size = 0.3
max_components = None
'''
# This is a sunny day test
#TODO add test_size as parameter of the function as the path directories for tests
def test_preprocess_1():
    max_comp=None
    test_size = 0.3
    normalize = True
    remove_outliers = True
    scalerType='StandardScaler'
    preprocess(norm, rm_outliers, scalerType='StandardScaler', max_comp=None)
'''
# This is a sunny day test
#TODO Randomize input and use test files
def test_preprocess_2(norm, rm_outliers, scalerType='StandardScaler', max_comp=None):
    max_comp=None
    test_size = 0.3
    normalize = True
    remove_outliers = True
    scalerType='StandardScaler'
    preprocess(norm, rm_outliers, scalerType='StandardScaler', max_comp=None)
'''
def test_features_selection_1():
    #TODO create random dataset with random dimensions
    randomDF=pd.DataFrame(np.random.rand(100, 3) , columns=list('XYZ'))
    n_components=3 # fail with n_components=5
    fs=features_selection(randomDF, n_components)
    assert (len(fs)==n_components)
test_features_selection_1()

'''
#TODO create random dataset with random dimensions
@given(dataset=dataset,n_components=integer)
def test_features_selection_2(dataset, n_components):
    randomDF=pd.DataFrame(np.random.rand(100, 20) , columns=list('XYZ'))
    fs=features_selection(dataset, n_components=5)
    assert (len(fs)==n_components)
'''
#TODO add the input from terminal to this test but how ???? 
def test_parse_arguments():
    parse_arguments()
    assert (True)

## this is the inspiration from the original documention on Hypothesis.works
'''

Values = st.integers()
SortedLists = st.lists(Values).map(sorted)

@given(ls=SortedLists, v=Values)
def test_insert_is_sorted(ls, v):
    """
    We test the first invariant: binary_search should return an index such
    that inserting the value provided at that index would result in a sorted
    set.
    """
    ls.insert(binary_search(ls, v), v)
    assert is_sorted(ls)
'''
