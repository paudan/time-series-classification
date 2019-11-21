from datetime import datetime
import os
import itertools
import pickle
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.transformers.shapelets import ContractedShapeletTransform
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import *

UCR_DATASET_PATH = '/mnt/DATA/data/Univariate_ts'
datasets = list(os.walk(UCR_DATASET_PATH))[0][1]
random_state = 0

classifiers = [RandomForestClassifier(n_estimators=100, random_state=random_state),
               ExtraTreesClassifier(n_estimators=100, random_state=random_state),
               XGBClassifier(random_state=random_state)]
tuples = ([(clf.__class__.__name__, 'Accuracy'), (clf.__class__.__name__, 'F1-Score')] for clf in classifiers)
index = pd.MultiIndex.from_tuples(itertools.chain(*tuples), names=['classifier', 'metric'])


def calculate_performance(output_file, excluded=None):
    excluded = excluded or []

    def evaluate_classifiers(dst):
        print("[%s] Processing dataset %s" % (datetime.now().strftime("%F %T"), dst))
        if dst in excluded:
            return None
        train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TRAIN.ts"))
        test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TEST.ts"))

        try:
            transform = ContractedShapeletTransform(time_limit_in_mins=5)
            train_data = transform.fit_transform(train_x, train_y)
            test_data = transform.transform(test_x)
        except:
            return None

        def evaluate_classifier(clf):
            try:
                clf.fit(train_data, train_y)
                pred = clf.predict(test_data)
                return accuracy_score(test_y, pred), f1_score(test_y, pred, average='macro')
            except:
                return float('nan'), float('nan')

        return list(itertools.chain(*[evaluate_classifier(clf) for clf in classifiers])), dst

    global datasets
    results = [evaluate_classifiers(dst) for dst in datasets]
    results = pd.DataFrame([x[0] for x in results if x is not None],
                           index=[x[1] for x in results if x is not None], columns=index)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    return results


if __name__ == '__main__':
    calculate_performance(os.path.join("results", "shapelets_results.pkl"),
                          excluded=['AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ'])