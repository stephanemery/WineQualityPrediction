import argparse
import pandas as pd
import numpy as np
import os
import ntpath
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

"""
Preprocessing functions
"""

def normalize(x, scalerType):
    """
    Nomalize the columns of the array passed

    Parameters
    ==========

    x : pandas.DataFrame
	The input data array (NxM).
    scalerType : sklearn.base.BaseEstimator
	Type of scaler to use (StandardScaler or MinMaxScaler).

    Returns
    =======
   
    : pandas.DataFrame
        The array normalized.

    """
    if len(x) > 1:
        result = np.zeros(x.shape)
        # Normalize data
        for i in range(x.shape[1]):
            # Select the column
            col = x.to_numpy()[:, i]
            # Normalize the column
            scaler = scalerType()
            result[:, i] = scaler.fit_transform(col.reshape(-1, 1)).squeeze()
        return pd.DataFrame(result, columns=x.columns, index=x.index)
    else:
        raise Exception("The dimension does not match the expected ones")


def remove_outliers(x):
    """
    Remove outliers (data outside of [1%, 99%]).

    Parameters
    ==========

    x : pandas.DataFrame
	The input data array (NxM).

    Returns
    =======
   
    : pandas.DataFrame
        The array without outliers.

    """
    result = x.copy()
    # Remove outliners
    for i in range(result.shape[1]):
        # Select the column
        col = result.iloc[:, i]
        # Find data between 1%-99%
        inLimits = col.between(col.quantile(0.01), col.quantile(0.99))
        # Remove the others
        result.drop(np.where(inLimits == False)[0], inplace=True)
        result.reset_index(drop=True, inplace=True)

    return result


def features_selection(dataset, n_components=5):
    """
    Select the most important features of the data.

    Parameters
    ==========

    dataset : pandas.DataFrame
	The input data array (NxM).
    n_components : int
	The number of features to keep.

    Returns
    =======
   
    : pandas.DataFrame
        A new array containing only the *n_components* most important features.

    """
    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(dataset)
    # Project axes in reduced space
    res = pca.transform(np.eye(dataset.shape[1]))
    # Compute contribution
    contrib = np.sum(abs(res), axis=1)
    # Sort features
    principal_features = np.argsort(contrib)

    return principal_features[-1 : -n_components - 1 : -1]


def preprocess(filepath, norm, rm_outliers, scalerType="StandardScaler", max_comp=None):    
    """
    Preprocess the data of the wine in the *data* folder.

    Parameters
    ==========

    norm : boolean
	Defined if the data have to be normalized.
    rm_outliers : boolean
	Defined if the outliers have to be removed.
    scalerType : sklearn.base.BaseEstimator
	Type of scaler to use (StandardScaler or MinMaxScaler).
    n_components : int
	The number of features to keep.

    """
    options = ""

    # Paths
    path_dir = ntpath.dirname(filepath)
    filename = ntpath.basename(filepath)  

    # Load data
    data = pd.read_csv(path_dir+'/'+filename, sep=";")

    # Drop NaN values
    data = data.dropna(axis="index")

    # Remove outliers
    if rm_outliers:
        data = remove_outliers(data)
        options += "ro_"

    # Normalize
    if norm:
        scalerType = StandardScaler if scalerType == "StandardScaler" else MinMaxScaler
        data = normalize(data, scalerType)
        options += "n_"

    if max_comp is not None:
        # Search for the X most contributing features
        princ_comp = features_selection(data.iloc[:, :-1], max_comp)
        # Add quality column
        princ_comp = np.append(princ_comp, -1)
        # Keep X principal components
        data = data.iloc[:, princ_comp]

    # Save
    data.to_csv(path_dir + "/preprocessed_" + options + filename, index=False)

    print("Preprocessing done !")
