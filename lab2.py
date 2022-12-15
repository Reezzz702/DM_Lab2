import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
import csv

def load(mode):
    data_root = "data/"
    data = pd.read_csv(data_root + mode + f"/{mode}_data.csv")
    if mode == 'train':
        label = pd.read_csv(data_root + mode + f"/{mode}_label.csv")
        return data, label
    return data, None


def upsample(data, label):
    data['has_died'] = label
    survived = data[data['has_died'] == 0]
    died = data[data['has_died'] == 1]

    died_upsampled = resample(died, n_samples=int(len(survived)/2))
    data_upsampled = pd.concat([survived, died_upsampled])
    # data_upsampled.groupby('has_died').size().plot(kind='pie',
    #                                    y = "v1",
    #                                    label = "Type",
    #                                    autopct='%1.1f%%')
    # plt.show()
    Y = data_upsampled['has_died']
    X = data_upsampled.drop('has_died', axis=1)
    return X,Y


def train_val():
    data, label = load("train")
    data = data.drop(['encounter_id', 'icu_id', 'patient_id', 'hospital_id'], axis=1)
    col = data.columns

    for c in data.columns:
            if data[c].dtype.name == "object":
                data[c] = pd.factorize(data[c], sort=True)[0].astype(int)

    data = preprocessing.scale(data)
    data = pd.DataFrame(data, columns=col)
    # label.groupby('has_died').size().plot(kind='pie',
    #                                    y = "v1",
    #                                    label = "Type",
    #                                    autopct='%1.1f%%')

    # plt.show()
    # plt.clf()

    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.3)
    X_upsampled, y_upsampled = upsample(X_train, y_train)

    xgb_model = xgb.XGBClassifier(objective="binary:logistic")
    xgb_model.fit(X_upsampled, y_upsampled, eval_set=[(X_val, y_val)], verbose=False)

    label_pred = xgb_model.predict(X_val)
    label_prob = xgb_model.predict_proba(X_val)[:, 1]

    print(confusion_matrix(y_val, label_pred))
    print(f1_score(y_val, label_pred, average='macro'))
    fpr, tpr, _ = roc_curve(y_val, label_prob)
    plt.plot(fpr,tpr)
    plt.show()
    xgb.plot_importance(xgb_model, max_num_features=20)
    plt.show()
    print(roc_auc_score(y_val, label_pred, average='macro'))

    return xgb_model

def test(model):
    data, _ = load("test")
    patient_id = data['patient_id']
    data = data.drop(['encounter_id', 'icu_id', 'patient_id', 'hospital_id'], axis=1)
    
    for c in data.columns:
            if data[c].dtype.name == "object":
                data[c] = pd.factorize(data[c], sort=True)[0].astype(int)
    data = preprocessing.scale(data)

    index = patient_id.argsort()
    pred = model.predict(data)

    with open("testing_result.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'pred'])
        for i in range(len(index)):
            writer.writerow([patient_id[index[i]], pred[index[i]]])
        
        f.close()

if __name__ == '__main__':
    model = train_val()
    test(model)