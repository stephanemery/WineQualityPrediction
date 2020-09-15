#https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
'''
Function which takes a 2D numpy array of features as input and return 
the ranking of the most relevant feature with their score 
'''
def PCA_function(dataset, n_components=2):
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(dataset)
    pcDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    #finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    print(pca.explained_variance_ratio_)

    print(pca.singular_values_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    return loadings
# Test to
if __name__ == "__main__":
    df= pd.read_csv("data\preprocessed_ro_n_red.csv")
    PCA_function(dataset=df)
    print(df.describe())

    print('Test function')


