import sys
import pandas as pd
from model.MultiLinearRegression import MultiLinearRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Preprocess
    # Remove outliers and normalize
    script_descriptor = open("./model/Preprocessing.py")
    script = script_descriptor.read()
    sys.argv = ["Preprocessing.py", "StandardScaler", "-n", "-ro"]
    # Run the preprocess script
    exec(script)
    script_descriptor.close()

    # Load data
    red_wine = pd.read_csv('./data/preprocessed_ro_n_red.csv')
    white_wine = pd.read_csv('./data/preprocessed_ro_n_white.csv')
    
    # Split into train/test dataset
    test_size = 0.3
    train_set, test_set = train_test_split(red_wine, test_size=test_size, shuffle=True)
    
    mlr = MultiLinearRegression()
    mlr.train(train_set.to_numpy()[:, :-1], train_set.to_numpy()[:, -1])
    mlr.test(test_set.to_numpy()[:, :-1], test_set.to_numpy()[:, -1])
    print(mlr)

    for i in range(10):
        a = mlr.predict(test_set.to_numpy()[i, :-1].reshape(1,-1))
        print(a, test_set.to_numpy()[i, -1])