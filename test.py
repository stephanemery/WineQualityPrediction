import numpy as np
import pandas as pd

from model.MultiLinearRegression import *
from model.Preprocessing import *

# from model import MultiLinearRegression
# from model import Preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import Machine
from hypothesis import given, strategies as st

Values = st.integers()
list1 = st.lists(Values)


@given(x=list1)
def test_normalize_1(x):
    x = np.array(x)
    if len(x) > 1:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        if std != 0:
            expected = (x - mean) / std
            result = normalize(pd.DataFrame(x), StandardScaler)
            np.testing.assert_array_almost_equal(
                expected, result.to_numpy().ravel(), decimal=6
            )


def test_remove_outliers_1():
    randomDF = pd.DataFrame(np.random.rand(100, 3), columns=list("XYZ"))
    to_compare = remove_outliers(randomDF)
    assert (
        randomDF.shape[0] * randomDF.shape[1]
        > to_compare.shape[0] * to_compare.shape[1]
    )


# create a random csv dataset with random values
Values = st.integers()


@given(x=Values)
def test_remove_outliers_2(x):
    print(x)
    x = int(abs(x))
    print(x)
    list1 = list(range(0, x))

    randomDF = pd.DataFrame(
        np.random.rand(100, len(list1)), columns=list1
    )  # list('XYZZ'))

    to_compare = remove_outliers(randomDF)
    assert (
        randomDF.shape[0] * randomDF.shape[1]
        > to_compare.shape[0] * to_compare.shape[1]
    )


# create a random csv dataset with only ones with different dimensions
Values = st.integers()


@given(x=Values)
def test_remove_outliers_3(x):
    print(x)
    x = int(abs(x))
    print(x)
    list1 = list(range(0, x))

    randomDF = pd.DataFrame(np.ones(100, len(list1)), columns=list1)  # list('XYZZ'))

    to_compare = remove_outliers(randomDF)
    assert (
        randomDF.shape[0] * randomDF.shape[1]
        > to_compare.shape[0] * to_compare.shape[1]
    )


test_remove_outliers_3()


# create a random csv dataset with only ones with different dimensions
Values = st.integers()


@given(x=Values)
def test_remove_outliers_3(x):
    print(x)
    x = int(abs(x))
    print(x)
    list1 = list(range(0, x))

    randomDF = pd.DataFrame(np.ones(100, len(list1)), columns=list1)  # list('XYZZ'))
    randomDF = df.mask(
        np.random.random(df.shape) < 0.1
    )  # Add Nan values in the dataset

    to_compare = remove_outliers(randomDF)
    assert (
        randomDF.shape[0] * randomDF.shape[1]
        == to_compare.shape[0] * to_compare.shape[1]
    )


test_remove_outliers_3()

"""
# Parameters
normalize = True
remove_outliers = True
scalerType='StandardScaler'
test_size = 0.3
max_components = None
"""
# This is a sunny day test
# TODO add test_size as parameter of the function and the path directories in preprocessing
def test_preprocess_1():
    max_comp = None
    test_size = 0.3
    norm = True
    rm_outliers = True
    scalerType = "StandardScaler"
    preprocess(norm, rm_outliers, scalerType="StandardScaler", max_comp=None)


"""
# This is a sunny day test
#TODO Randomize input and use test files
def test_preprocess_2(norm, rm_outliers, scalerType='StandardScaler', max_comp=None):
    max_comp=None
    test_size = 0.3
    normalize = True
    remove_outliers = True
    scalerType='StandardScaler'
    preprocess(norm, rm_outliers, scalerType='StandardScaler', max_comp=None)
"""


def test_features_selection_1():
    # TODO create random dataset with random dimensions
    randomDF = pd.DataFrame(np.random.rand(100, 3), columns=list("XYZ"))
    n_components = 3  # fail with n_components=5
    fs = features_selection(randomDF, n_components)
    assert len(fs) == n_components


"""
from hypothesis.extra.pandas import column, data_frames
#create a random dataset with random dimensions
dataset=data_frames([column('A', dtype=int), column('B', dtype=float)]).example()
@given(dataset=dataset,n_components=Values)
def test_features_selection_2(dataset, n_components):
    
    randomDF=pd.DataFrame(np.random.rand(100, 20) , columns=list('XYZ'))
    fs=features_selection(randomDF, n_components=5)
    assert (len(fs)==n_components)

#TODO add  input from terminal to this test but how ???? 
def test_parse_arguments():
    parse_arguments()
    assert (True)
"""


def test_machine_1():
    mach1 = Machine("KNN Regressor")
    mach1.train()
    mach1.test()
    MultiLinearRegression(mach1)


"""
Integration test of the main function which needs to run all the functions
"""
"""
def test_main():
    main()
"""

"""
function to test if the test works as expected
"""

print("test_normalize_1")
test_normalize_1()
print("test_remove_outliers_1")
test_remove_outliers_1()
# print('test_remove_outliers_2')
# test_remove_outliers_2()
print("test_features_selection_1")
test_features_selection_1()
# print('test_features_selection_2')
# test_features_selection_2()
print("test_preprocess_1")
test_preprocess_1()
# print('test_preprocess_2')
# test_preprocess_2()
# print('test_parse_arguments')
# test_parse_arguments()

"""
print('test_main')
test_main()
"""
