import sys
import pandas as pd
import numpy as np
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

def remove_outliners(x):
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

def preprocess(scalerType):
    # Load data
    red_wine = pd.read_csv('../data/winequality-red.csv')
    white_wine = pd.read_csv('../data/winequality-white.csv')

    # Drop NaN values
    red_wine = red_wine.dropna(axis='index')
    white_wine = white_wine.dropna(axis='index')
    
    # Remove outliers
    red_wine = remove_outliners(red_wine)
    white_wine = remove_outliners(white_wine)

    # Normalize 
    red_wine = normalize(red_wine)
    white_wine = normalize(white_wine)

    # Save
    red_wine.to_csv('../data/normalized_red.csv',index=False)
    white_wine.to_csv('../data/normalized_white.csv',index=False)

    print('Preprocessing done !')


if __name__ == "__main__":    
    if len(sys.argv) < 2 or sys.argv[1] == '-h':
        print('Usage : Preprocessing.py [-h]')
        print('\r\npositional arguments:\r\n\t\tThe name of the scaler : "StandardScaler", "MinMaxScaler"')
        sys.exit()

    if sys.argv[1] != 'StandardScaler' and sys.argv[1] != 'MinMaxScaler':
        print('You should select "StandardScaler" or "MinMaxScaler"')
        sys.exit()

    print('Preprocessing...')

    scalerType = StandardScaler if sys.argv[1] == 'StandardScaler' else MinMaxScaler
    preprocess(scalerType)
    