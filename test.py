import nose.tools as nt
import numpy as np
import pandas as pd

from model.Preprocessing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hypothesis import given, strategies as st
Values = st.integers()

list1 = st.lists(Values)
array1=np.array(list1)

@given(x=array1)
def test_normalize(x):
    x = np.array(x)
    #x = np.random.random((5,3))
    mean = np.mean(x, axis=0)
    std = np.std  (x, axis=0)
    expected =    (x-mean)/std 
    result = normalize(pd.DataFrame(x), StandardScaler)
    np.testing.assert_array_almost_equal(expected, result, decimal=6)

'''
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