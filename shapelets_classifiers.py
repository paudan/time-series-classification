from datetime import datetime
import os
import itertools
import pickle
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.transformers.shapelets import ContractedShapeletTransform
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import *

UCR_DATASET_PATH = '/mnt/DATA/data/Univariate_ts'
datasets = list(os.walk(UCR_DATASET_PATH))[0][1]
random_state = 0

classifiers = [RandomForestClassifier(n_estimators=100, random_state=random_state),
               ExtraTreesClassifier(n_estimators=100, random_state=random_state),
               LGBMClassifier(random_state=random_state)]
tuples = ([(clf.__class__.__name__, 'Accuracy'), (clf.__class__.__name__, 'F1-Score')] for clf in classifiers)
index = pd.MultiIndex.from_tuples(itertools.chain(*tuples), names=['classifier', 'metric'])


def calculate_performance(output_file):

    def evaluate_classifiers(dst):
        print("[%s] Processing dataset %s" % (datetime.now().strftime("%F %T"), dst))

        train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TRAIN.ts"))
        test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TEST.ts"))

        transform = ContractedShapeletTransform(time_limit_in_mins=5)
        train_data = transform.fit_transform(train_x, train_y)
        test_data = transform.transform(test_x)

        def evaluate_classifier(clf):
            clf.fit(train_data, train_y)
            pred = clf.predict(test_data)
            return accuracy_score(test_y, pred), f1_score(test_y, pred, average='macro')

        return list(itertools.chain(*[evaluate_classifier(clf) for clf in classifiers]))

    global datasets
    datasets = datasets[:2]
    results = pd.DataFrame([evaluate_classifiers(dst) for dst in datasets], index=datasets, columns=index)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    return results


if __name__ == '__main__':
    print(calculate_performance(os.path.join("results", "shapelets_results.pkl")))