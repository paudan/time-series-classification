from datetime import datetime
import os
import itertools
import pickle
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import *
from catch22 import catch22_all
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

UCR_DATASET_PATH = '/mnt/DATA/data/Univariate_ts'
datasets = list(os.walk(UCR_DATASET_PATH))[0][1]
random_state = 0

classifiers = [RandomForestClassifier(n_estimators=100, random_state=random_state),
               ExtraTreesClassifier(n_estimators=100, random_state=random_state),
               XGBClassifier(random_state=random_state)]
tuples = ([(clf.__class__.__name__, 'Accuracy'), (clf.__class__.__name__, 'F1-Score')] for clf in classifiers)
index = pd.MultiIndex.from_tuples(itertools.chain(*tuples), names=['classifier', 'metric'])


def tsfresh_features(series_data):
    series_df = pd.concat([pd.DataFrame({'id': i, 'value': series_data.iloc[i][0]}).reset_index()
                           for i in range(series_data.shape[0])], axis='rows')
    series_df.columns = ['time', 'id', 'value']
    f = extract_features(series_df, column_id = "id", column_sort = "time")
    impute(f)
    return f

def catch22_features(series_data):
    def series_features(series):
        feat = catch22_all(series)
        return dict(zip(feat['names'], feat['values']))

    return pd.DataFrame([series_features(series_data.iloc[i][0]) for i in range(series_data.shape[0])])


def calculate_performance(func, output_file, excluded=None):
    excluded = excluded or []

    def evaluate_classifiers(dst):
        print("[%s] Processing dataset %s" % (datetime.now().strftime("%F %T"), dst))
        if dst in excluded:
            return None
        train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TRAIN.ts"))
        test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TEST.ts"))
        if len(set(train_y)) == 1:
            return None
        data_train = func(train_x)
        data_test = func(test_x)

        def evaluate_classifier(clf):
            try:
                clf.fit(data_train, train_y)
                pred = clf.predict(data_test)
                return accuracy_score(test_y, pred), f1_score(test_y, pred, average='macro')
            except:
                return float('nan'), float('nan')

        return list(itertools.chain(*[evaluate_classifier(clf) for clf in classifiers]))

    global datasets
    #datasets = datasets[:20]
    results = [evaluate_classifiers(dst) for dst in datasets]
    results = pd.DataFrame([x for x in results if x is not None],
                           index=[d for d in datasets if d not in excluded], columns=index)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    return results


if __name__ == '__main__':
#    calculate_performance(tsfresh_features, os.path.join("results", "tsfresh_results.pkl"))
    calculate_performance(catch22_features, os.path.join("results", "catch22_results.pkl"),
                          excluded=['AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ'])  # Segfault