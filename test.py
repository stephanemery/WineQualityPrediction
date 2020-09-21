import nose.tools as nt
import numpy as np
import pandas as pd

from model.Preprocessing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def test_normalize():
    x = np.random.random((5,3))
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    expected = (x-mean)/std 
    result = normalize(pd.DataFrame(x), StandardScaler)

    np.testing.assert_array_almost_equal(expected, result, decimal=6)
