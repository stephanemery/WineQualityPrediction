import argparse
import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize(x):
    result = np.zeros(x.shape)
    # Normalize data
    for i in range(x.shape[1]):
        # Select the column
        col = x.iloc[:, i].to_numpy()
        # Normalize the column
        scaler = scalerType()        
        result[:, i] = scaler.fit_transform(col.reshape(-1,1)).squeeze()

    return pd.DataFrame(result, columns=x.columns, index=x.index)

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

def preprocess(args, scalerType):
    options = ''

    # Paths
    path_dir = './data'#os.path.dirname(os.path.abspath(__file__))
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
    if args.remove_outliers:
        red_wine = remove_outliers(red_wine)
        white_wine = remove_outliers(white_wine)
        options += 'ro_'

    # Normalize 
    if args.normalize:
        red_wine = normalize(red_wine)
        white_wine = normalize(white_wine)
        options += 'n_'

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

    scalerType = StandardScaler if args.scaler == 'StandardScaler' else MinMaxScaler
    preprocess(args, scalerType)
    