import argparse
import sys

from wineQualityPred.paper import predictQuality
from wineQualityPred.paper import reproduceResults


def parse_arguments(args):
    '''
    Parse the arguments of the command line
    '''
    parser = argparse.ArgumentParser(description="Predict wine quality from its physicochemical properties.")
    parser.add_argument( "-f",
        "--filepath",
        type=str,
        help='Filepath of the data to process.', default=None
    )
    parser.add_argument( "-s",
        "--scaler",
        type=str,
        help='The name of the scaler : "StandardScaler", "MinMaxScaler"', default="StandardScaler"
    )
    parser.add_argument("-nn", "--not_normalize", help="Do not normalize data", action="store_true")
    parser.add_argument("-ns", "--not_shuffle", help="Do not shuffle data", action="store_true")
    parser.add_argument(
        "-nro", "--not_remove_outliers", help="Do not remove outliers", action="store_true"
    )

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])   

    try:
        filepath = args.filepath
        if filepath is None :
            filepath ='wineQualityPred/data/winequality-red.csv'

        predictQuality(filepath, not args.not_shuffle, not args.not_normalize, not args.not_remove_outliers, args.scaler)
    except Exception as e:
        print(e)
