import nose.tools as nt
import numpy as np
import pandas as pd

from model.MultiLinearRegression import *
from model.Preprocessing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from hypothesis import given, strategies as st

Values = st.integers()
list1 = st.lists(Values)
#array1=np.array(list1)

@given(x=list1)
def test_normalize(x):
    x = np.array(x)
    if len(x)>1 :
        #x = np.random.random((5,3))
        mean = np.mean(x, axis=0)
        std = np.std  (x, axis=0)
        if(std!=0):
            expected =    (x-mean)/std 
            print(x)
            result = normalize(pd.DataFrame(x), StandardScaler)
            np.testing.assert_array_almost_equal(expected, result, decimal=np.array().size)
            print(x)
# Test a set of test functions
try:
    test_normalize()
except AssertionError:
    print('Not Defined')


Values = st.integers()
list1 = st.lists(Values)  
@given(x=list1)
def remove_outliers(x):
    x = np.array(x)

    assert()
 

def test_preprocess(norm, rm_outliers, scalerType='StandardScaler', max_comp=None):
    assert()


Values = st.integers()
SortedLists = st.lists(Values).map(sorted)

def test_normalize():

@given(ls=SortedLists, v=Values)
def test_insert_is_sorted(ls, v):
    """We test the first invariant: binary_search should return an index such
    that inserting the value provided at that index would result in a sorted
    set."""
    ls.insert(binary_search(ls, v), v)
    assert is_sorted(ls)
'''