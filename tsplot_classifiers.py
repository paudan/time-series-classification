import os
import itertools
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from skimage.transform import resize
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import *
from pyts.image import RecurrencePlot, MarkovTransitionField, GramianAngularField
from keras.layers import Conv2D, InputLayer, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from keras.models import Sequential
from keras.utils import Sequence


input_dim = 128
n_epochs = 10  # 5 at least
batch_size = 16

UCR_DATASET_PATH = '/mnt/DATA/data/Univariate_ts'
datasets = list(os.walk(UCR_DATASET_PATH))[0][1]
random_state = 0

classifiers = ["RecurrencePlot", "MarkovTransitiveFields", "GramianAngular"]
tuples = ([(clf, 'Accuracy'), (clf, 'F1-Score')] for clf in classifiers)
index = pd.MultiIndex.from_tuples(itertools.chain(*tuples), names=['classifier', 'metric'])


class SeriesPlotGenerator(Sequence):

    def __init__(self, series, labels, img_size=128, batch_size=32, series_plot_obj=None):
        self.series = series
        self.labels = labels
        self.img_dim = img_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.series))
        self.rp = series_plot_obj
        if self.rp is None:
            self.rp = RecurrencePlot(threshold='point', percentage=20)

    @staticmethod
    def generate_series_plot(rp, data, img_dim):
        im_plot = rp.fit_transform([data])
        im_resized = resize(im_plot[0], (img_dim, img_dim), mode='constant', preserve_range=True)
        im_resized = np.expand_dims(im_resized, axis=2)    # Add channel dim
        return im_resized

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([self.generate_series_plot(self.rp, self.series[idx], self.img_dim) for idx in indexes]), self.labels[indexes]

    def __len__(self):
        return int(np.floor(len(self.series)/ self.batch_size))


class TimeSeriesPlotClassifier:

    def __init__(self, input_dim, num_classes, batch_size=16, series_plot_obj=None):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.series_plot_obj = series_plot_obj
        if self.series_plot_obj is None:
            self.series_plot_obj = RecurrencePlot(threshold='point', percentage=20)
        self.init_model()


    def init_model(self):
        self.model = Sequential()
        input = InputLayer(input_shape=(self.input_dim, self.input_dim, 1))
        self.model.add(input)
        # encoder
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes))
        self.model.compile(loss='mse', optimizer='adam')
        self.model.summary()

    def train(self, series_data, labels, n_epochs=5):
        plot_generator = SeriesPlotGenerator(series_data, labels, img_size=self.input_dim,
                                             batch_size=self.batch_size, series_plot_obj=self.series_plot_obj)
        return self.model.fit_generator(plot_generator, epochs=n_epochs)

    def predict(self, data):
        img = SeriesPlotGenerator.generate_series_plot(self.series_plot_obj, data, self.input_dim)
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return self.model.predict_classes(img)


def calculate_performance(output_file):

    def evaluate_classifiers(dst):
        print("[%s] Processing dataset %s" % (datetime.now().strftime("%F %T"), dst))

        train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TRAIN.ts"))
        test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TEST.ts"))
        data_train = [train_x.iloc[i][0] for i in range(train_x.shape[0])]
        data_test = [test_x.iloc[i][0] for i in range(test_x.shape[0])]
        enc = LabelEncoder().fit(train_y)
        ohe = OneHotEncoder(sparse=False)
        labels_encoded = enc.transform(train_y)
        integer_encoded = labels_encoded.reshape(len(labels_encoded), 1)
        labels_train = ohe.fit_transform(integer_encoded)
        ts_plotters = [RecurrencePlot(threshold='point', percentage=20),
                       MarkovTransitionField(),
                       GramianAngularField()]

        def evaluate_classifier(plot_obj):
            try:
                classifier = TimeSeriesPlotClassifier(input_dim, num_classes=len(set(train_y)),
                                                      batch_size=batch_size, series_plot_obj=plot_obj)
                classifier.train(data_train, labels_train, n_epochs=n_epochs)
                y_pred = [classifier.predict(series) for series in data_test]
                y_pred = enc.inverse_transform(y_pred)
                return accuracy_score(test_y, y_pred), f1_score(test_y, y_pred, average='macro')
            except:
                return float('nan'), float('nan')

        return list(itertools.chain(*[evaluate_classifier(plot_obj) for plot_obj in ts_plotters]))

    global datasets
    results = pd.DataFrame([evaluate_classifiers(dst) for dst in datasets], index=datasets, columns=index)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    return results


if __name__ == '__main__':
    calculate_performance(os.path.join("results", "cnn_results.pkl"))
