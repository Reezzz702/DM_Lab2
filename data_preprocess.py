import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from argparse import ArgumentParser

def load(mode):
    data_root = "data/"
    data = pd.read_csv(data_root + mode + f"/{mode}_data.csv")
    if mode == 'train':
        label = pd.read_csv(data_root + mode + f"/{mode}_label.csv")
        return data, label
    return data, None    


def train_val(args):
    data, label = load(args.mode)
    data = data.drop(['encounter_id', 'icu_id', 'patient_id', 'hospital_id'], axis=1)
    for c in data.columns:
            if data[c].dtype.name == "object":
                data[c] = pd.factorize(data[c], sort=True)[0].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.3)

    best_model = None
    xgb_model = xgb.XGBClassifier(objective="binary:logistic")
    xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)

    xgb.plot_importance(xgb_model, max_num_features=20)
    plt.show()
    # check missing value count
    # print(data.isnull().sum(axis=0).sort_values(ascending=False))
    # data = data[data.isna().sum(axis=1) == 0]
    label_pred = xgb_model.predict(X_val)

    print(confusion_matrix(y_val, label_pred))
    print(f1_score(y_val, label_pred, average='macro'))
    print(roc_auc_score(y_val, label_pred, average='macro'))

    return best_model

def test(args, model):
    data, _ = load(args.mode)
    test_data = data.drop(['encounter_id', 'icu_id', 'patient_id', 'hospital_id'], axis=1)
    
    for c in data.columns:
            if data[c].dtype.name == "object":
                data[c] = pd.factorize(data[c], sort=True)[0].astype(int)

    pred = model.predict(data)

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-m", dest="mode", help="train or test set", type=str
    )
    best_model = train_val(parser.parse_args())
    test(parser.parse_args(), best_model)