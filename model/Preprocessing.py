import argparse
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
'''

'''
def normalize(x, scalerType):
    if len(x)>1 :
        result = np.zeros(x.shape)
        # Normalize data
        for i in range(x.shape[1]):
            # Select the column
            col = x.to_numpy()[:, i]
            # Normalize the column
            scaler = scalerType()        
            result[:, i] = scaler.fit_transform(col.reshape(-1,1)).squeeze()
        return pd.DataFrame(result, columns=x.columns, index=x.index)
    else:
        print('The dimension does not match the expected ones')
        raise
    

def remove_outliers(x):
    result = x.copy()
    # Remove outliners
    for i in range(result.shape[1]):
        # Select the column
        col = result.iloc[:,i]
        # Find data between 1%-99%
        inLimits = col.between(col.quantile(.01), col.quantile(.99))
        # Remove the others
        result.drop(np.where(inLimits==False)[0], inplace=True)
        result.reset_index(drop=True, inplace=True)     

    return result

def features_selection(dataset, n_components=5):
    n_components = n_components//1
    if n_components>1: #and the shape is 2 dimensional and greater than (0,x) (x,0)
        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(dataset)
        # Project axes in reduced space
        res = pca.transform(np.eye(dataset.shape[1]))
        # Compute contribution
        contrib = np.sum(abs(res), axis=1)
        # Sort features
        principal_features = np.argsort(contrib)

        return principal_features[-1:-n_components-1:-1]

def preprocess(norm, rm_outliers, scalerType='StandardScaler', max_comp=None):
    options = ''

    # Paths
    path_dir = './data'
    path_red_wine = path_dir + '/winequality-red.csv'
    path_white_wine = path_dir + '/winequality-white.csv'
    
    # If data files don't exist, download them
    # Red wine data
    if not os.path.isfile(path_red_wine):
        print('Downloading red wine data...')
        my_file = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
        open(path_red_wine, 'wb').write(my_file.content)
    # White wine data
    if not os.path.isfile(path_white_wine):
        print('Downloading white wine data...')
        my_file = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
        open(path_white_wine, 'wb').write(my_file.content)

    # Load data
    red_wine = pd.read_csv(path_red_wine, sep=';')
    white_wine = pd.read_csv(path_white_wine, sep=';')

    # Drop NaN values
    red_wine = red_wine.dropna(axis='index')
    white_wine = white_wine.dropna(axis='index')
    
    # Remove outliers
    if rm_outliers:
        red_wine = remove_outliers(red_wine)
        white_wine = remove_outliers(white_wine)
        options += 'ro_'

    # Normalize 
    if norm:
        scalerType = StandardScaler if scalerType == 'StandardScaler' else MinMaxScaler
        red_wine = normalize(red_wine, scalerType)
        white_wine = normalize(white_wine, scalerType)
        options += 'n_'

    if max_comp is not None :
        # Search for the X most contributing features
        princ_comp_red = features_selection(red_wine.iloc[:,:-1], max_comp)
        princ_comp_white = features_selection(white_wine.iloc[:,:-1], max_comp)
        # Add quality column
        princ_comp_red = np.append(princ_comp_red, -1)
        princ_comp_white = np.append(princ_comp_white, -1)
        # Keep X principal components
        red_wine = red_wine.iloc[:, princ_comp_red]
        white_wine = white_wine.iloc[:, princ_comp_white]

    # Save
    red_wine.to_csv(path_dir + '/preprocessed_'+options+'red.csv',index=False)
    white_wine.to_csv(path_dir + '/preprocessed_'+options+'white.csv',index=False)

    print('Preprocessing done !')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess wine data.')
    parser.add_argument('scaler', type=str, help='The name of the scaler : "StandardScaler", "MinMaxScaler"')
    parser.add_argument('-n','--normalize', help='Normalize data', action='store_true')
    parser.add_argument('-ro','--remove_outliers', help='Remove outliers',action='store_true')

    return parser.parse_args()

if __name__ == "__main__":   
    args = parse_arguments()
    
    print('Preprocessing...')

    preprocess(args.normalize, args.remove_outliers, args.scaler)
