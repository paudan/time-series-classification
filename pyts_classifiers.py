from datetime import datetime
import os
import itertools
import pickle
import pandas as pd
from pyts.classification import BOSSVS, SAXVSM
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.metrics import *

UCR_DATASET_PATH = '/mnt/DATA/data/Univariate_ts'
datasets = list(os.walk(UCR_DATASET_PATH))[0][1]
random_state = 0

classifiers = [SAXVSM(n_bins=4, strategy='uniform', window_size=2, sublinear_tf=True),
               BOSSVS(word_size=2, n_bins=4, window_size=2)]
tuples = ([(clf.__class__.__name__, 'Accuracy'), (clf.__class__.__name__, 'F1-Score')] for clf in classifiers)
index = pd.MultiIndex.from_tuples(itertools.chain(*tuples), names=['classifier', 'metric'])


def calculate_performance(output_file):

    def evaluate_classifiers(dst):
        print("[%s] Processing dataset %s" % (datetime.now().strftime("%F %T"), dst))

        train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TRAIN.ts"))
        test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TEST.ts"))
        data_train = [train_x.iloc[i][0] for i in range(train_x.shape[0])]
        data_test = [test_x.iloc[i][0] for i in range(test_x.shape[0])]

        def evaluate_classifier(clf):
            clf.fit(data_train, train_y)
            pred = clf.predict(data_test)
            return accuracy_score(test_y, pred), f1_score(test_y, pred, average='macro')

        return list(itertools.chain(*[evaluate_classifier(clf) for clf in classifiers]))

    global datasets
    datasets = datasets[:2]
    results = pd.DataFrame([evaluate_classifiers(dst) for dst in datasets], index=datasets, columns=index)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    return results