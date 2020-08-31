import argparse
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

    # Load data
    red_wine = pd.read_csv('../data/winequality-red.csv')
    white_wine = pd.read_csv('../data/winequality-white.csv')

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
    red_wine.to_csv('../data/preprocessed_'+options+'red.csv',index=False)
    white_wine.to_csv('../data/preprocessed_'+options+'white.csv',index=False)

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
    