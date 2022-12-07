import pandas as pd
import numpy as np
from argparse import ArgumentParser

def load(mode):
    data_root = "data/"
    data = pd.read_csv(data_root + mode + f"/{mode}_data.csv")
    if mode == 'train':
        label = pd.read_csv(data_root + mode + f"/{mode}_label.csv")
        return data, label
    return data, None



def check_type(df):
    for c in df.columns:
        print(df.c.dtype)


def main(args):
    data, label = load(args.mode)
    print(data.info())
    check_type(data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-m", dest="mode", help="train or test set", type=str
    )
    main(parser.parse_args())