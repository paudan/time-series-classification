from datetime import datetime
import os
import itertools
import pickle
import numpy as np
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.transformers.compose import RowwiseTransformer
from sktime.transformers.compose import ColumnTransformer
from sktime.transformers.compose import Tabulariser
from sktime.transformers.segment import RandomIntervalSegmenter
from sklearn.preprocessing import FunctionTransformer
from sktime.pipeline import Pipeline
from sktime.pipeline import FeatureUnion
from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.classifiers.distance_based import ProximityForest
from sktime.classifiers.dictionary_based import BOSSEnsemble
from sktime.classifiers.frequency_based import RandomIntervalSpectralForest
from sktime.classifiers.interval_based import TimeSeriesForest
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

UCR_DATASET_PATH = '/mnt/DATA/data/Univariate_ts'
datasets = list(os.walk(UCR_DATASET_PATH))[0][1]
random_state = 0


def ar_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    model = AR(endog=x)
    return model.fit(maxlag=nlags).params.ravel()

def acf_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    return acf(x, nlags=nlags).ravel()

def powerspectrum(x, **kwargs):
    x = np.asarray(x).ravel()
    fft = np.fft.fft(x)
    ps = fft.real * fft.real + fft.imag * fft.imag
    return ps[:ps.shape[0] // 2].ravel()


rise_steps = [
    ('segment', RandomIntervalSegmenter(n_intervals=1, min_length=5)),
    ('transform', FeatureUnion([
        ('ar', RowwiseTransformer(FunctionTransformer(func=ar_coefs, validate=False))),
        ('acf', RowwiseTransformer(FunctionTransformer(func=acf_coefs, validate=False))),
        ('ps', RowwiseTransformer(FunctionTransformer(func=powerspectrum, validate=False)))
    ])),
    ('tabularise', Tabulariser()),
    ('clf', DecisionTreeClassifier())
]
base_estimator = Pipeline(rise_steps)
# ('RISE', TimeSeriesForestClassifier(base_estimator=base_estimator, n_estimators=100, bootstrap=True)),
classifiers = [('TimeSeriesForest', TimeSeriesForest()),
               ('ProximityForest', ProximityForest(n_trees=100)),
               ('BOSS', BOSSEnsemble()),
               ('RandomIntervalSpectralForest', RandomIntervalSpectralForest())]
tuples = ([(name, 'Accuracy'), (name, 'F1-Score')] for name, _ in classifiers)
index = pd.MultiIndex.from_tuples(itertools.chain(*tuples), names=['classifier', 'metric'])


def calculate_performance(output_file):

    def evaluate_classifiers(dst):
        print("[%s] Processing dataset %s" % (datetime.now().strftime("%F %T"), dst))

        train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TRAIN.ts"))
        test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TEST.ts"))

        def evaluate_classifier(clf):
            clf.fit(train_x, train_y)
            pred = clf.predict(test_x)
            return accuracy_score(test_y, pred), f1_score(test_y, pred, average='macro')

        return list(itertools.chain(*[evaluate_classifier(clf) for _, clf in classifiers]))

    global datasets
    datasets = datasets[:2]
    results = pd.DataFrame([evaluate_classifiers(dst) for dst in datasets], index=datasets, columns=index)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    return results


if __name__ == '__main__':
    print(calculate_performance(os.path.join("results", "sktime_results.pkl")))