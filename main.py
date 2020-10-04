import sys
import pandas as pd
import argparse

from model.Preprocessing import preprocess
from model.MultiLinearRegression import MultiLinearRegression
from model.KNN import KNN
from model.SVM import SVM
from sklearn.model_selection import train_test_split

def main(shuffle=True, normalize=True, remove_outliers=True, scalerType="StandardScaler", test_size=0.3, max_components=None):
    """
    Preprocess the data and try diffent model of learning on it.
    The function print the score of each model.
    """
    # Preprocess
    preprocess(normalize, remove_outliers, scalerType, max_components)

    # Load data
    options = "_"
    if remove_outliers:
        options += "ro_"
    if normalize:
        options += "n_"

    red_wine = pd.read_csv("./data/preprocessed" + options + "red.csv")
    white_wine = pd.read_csv("./data/preprocessed" + options + "white.csv")

    # Split into train/test dataset
    red_train_set, red_test_set = train_test_split(
        red_wine, test_size=test_size, shuffle=shuffle
    )
    white_train_set, white_test_set = train_test_split(
        white_wine, test_size=test_size, shuffle=shuffle
    )

    # Models
    red_models = []
    white_models = []
    # Add multi linear regression
    red_models.append(MultiLinearRegression())
    white_models.append(MultiLinearRegression())
    # Add KNN regressor
    red_models.append(KNN())
    white_models.append(KNN())
    # Add SVM regressor
    red_models.append(SVM(0, 0.5))
    white_models.append(SVM(0, 0.5))

    # Train red wine model
    for m in red_models:
        m.train(red_train_set.iloc[:, :-1], red_train_set.iloc[:, -1])

    # Train white wine model
    for m in white_models:
        m.train(white_train_set.iloc[:, :-1], white_train_set.iloc[:, -1])

    # Test red wine model
    for m in red_models:
        m.test(red_test_set.iloc()[:, :-1], red_test_set.iloc()[:, -1])

    # Test white wine model
    for m in white_models:
        m.test(white_test_set.iloc()[:, :-1], white_test_set.iloc()[:, -1])

    # Print scores for red wine
    print("Score for the red wine :")
    for m in red_models:
        print(m)

    print("")

    # Print scores for white wine
    print("Score for the white wine :")
    for m in white_models:
        print(m)



def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict wine quality from its physicochemical properties.")
    parser.add_argument(
        "--scaler",
        type=str,
        help='The name of the scaler : "StandardScaler", "MinMaxScaler"', default="StandardScaler"
    )
    parser.add_argument("-nn", "--not_normalize", help="Do not normalize data", action="store_true")
    parser.add_argument("-ns", "--not_shuffle", help="Do not shuffle data", action="store_true")
    parser.add_argument(
        "-nro", "--not_remove_outliers", help="Do not remove outliers", action="store_true"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    main(not args.not_shuffle, not args.not_normalize, not args.not_remove_outliers, args.scaler)