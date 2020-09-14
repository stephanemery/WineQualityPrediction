import sys
import pandas as pd
from model.MultiLinearRegression import MultiLinearRegression
from model.KNN import KNN
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
    
    # Models
    models = []
    # Add multi linear regression
    models.append(MultiLinearRegression())
    # Add KNN regressor
    models.append(KNN())

    # Train
    for m in models:
        m.train(train_set.to_numpy()[:, :-1], train_set.to_numpy()[:, -1])

    # Test 
    for m in models:
        m.test(test_set.to_numpy()[:, :-1], test_set.to_numpy()[:, -1])

    # Print scores
    for m in models:
        print(m)    
