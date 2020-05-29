import argparse
import time
import numpy as np
import tensorflow as tf
import sys
import json
import os.path as op

from numpy import random
from keras.utils import to_categorical

from elisa_mlayer import (
    config,
    utils
)
from elisa_mlayer.gen import conf
from elisa_mlayer.logger import getLogger
from elisa_mlayer.nn.clsf import base
from elisa_mlayer.sio import SyntheticFlatMySqlIO
from elisa_mlayer.nn.base import layers, losses, nn, optimizers
from elisa_mlayer.nn.clsf.base import KerasNet

random.seed(int(time.time()))
logger = getLogger("nn.clsf.mlp.has_spots")


class Feed(SyntheticFlatMySqlIO):
    def get_feed(self, test_size=0.05, threshold=0.05, relative_threshold=True, passband='Generic.Bessell.V'):
        logger.info('initializing db session')
        self._preinit_method()
        session = self._get_session()

        logger.info('getting training data')
        data = session.query(self._model_instance) \
            .with_entities(
            self._model_declarative_meta.spotty,
            getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
        ).all()

        self.finish_session(session, w=False)

        logger.info('preparing test and training batches')
        ys, xs = zip(*data)
        ys, xs = np.array(ys), np.array([json.loads(row) for row in xs])
        xs = np.divide(xs, xs.max(axis=1)[:, None])

        # apply threshold
        if relative_threshold:
            r_diff = np.abs((xs[:, 28][ys] - xs[:, 70][ys])) / (xs[ys].max(axis=1) - xs[ys].min(axis=1))
        else:
            r_diff = np.abs(xs[:, 28][ys] - xs[:, 70][ys])
        oconell = np.greater(r_diff, threshold)

        # if len(ys[ys][oconell]) < len(ys[ys]) * 0.5:
        #     raise ValueError("Value `flux-threshold` is too aggressive")

        xs = np.concatenate((xs[~ys], xs[ys][oconell]))
        ys = np.concatenate((ys[~ys], ys[ys][oconell]))

        logger.info('shuffling data')
        idx = np.arange(0, len(xs), 1)
        np.random.shuffle(idx)

        xs = xs[idx]
        ys = ys[idx]

        all_entities = len(xs)
        test_samples_xs, test_samples_ys = xs[:int(all_entities * test_size)], ys[:int(all_entities * test_size)]
        train_samples_xs, train_samples_ys = xs[int(all_entities * test_size):], ys[int(all_entities * test_size):]

        train_samples_ys = self.parse_y_feed(train_samples_ys)
        test_samples_ys = self.parse_y_feed(test_samples_ys)

        return utils.convert_to_numpy_array((train_samples_xs, train_samples_ys, test_samples_xs, test_samples_ys))

    @classmethod
    def parse_y_feed(cls, y_data):
        return np.array(y_data, dtype=int)


class AbstractHasSpotsNet(KerasNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size=test_size, **kwargs)

        self._n_class = 2
        self._passband = str(passband)
        self._table_name = str(kwargs.get("table_name", "synthetic_lc"))
        self._flux_threshold = float(kwargs.get("flux_threshold", 0.0))
        self._relative_threshold = bool(kwargs.get("relative_threshold", True))

        if not self._from_pickle and self._reinitialize_feed:
            logger.info("obtaining training data")

            self._feed = Feed(config.DB_CONF, table_name=self._table_name)
            self.train_xs, self.train_ys, self.test_xs, self.test_ys = \
                self._feed.get_feed(test_size=test_size, threshold=self._flux_threshold,
                                    relative_threshold=self._relative_threshold, passband=passband)


class MlpNet(AbstractHasSpotsNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size, passband, **kwargs)

        if not self._from_pickle and self._reinitialize_feed:
            self.train_ys = to_categorical(self.train_ys, self._n_class)
            self.test_ys = to_categorical(self.test_ys, self._n_class)

        logger.info("creating neural model")
        self.model = tf.keras.Sequential()
        vector_size = 100
        self.model.add(layers.Dense(128, activation=nn.tanh, input_shape=(vector_size, )))
        self.model.add(layers.Dense(256, activation=nn.tanh))
        self.model.add(layers.Dense(128, activation=nn.tanh))
        self.model.add(layers.Dense(2, activation=nn.softmax))

        optimizer = optimizers.Adam(lr=self._learning_rate, decay=self._optimizer_decay)
        loss_fn = losses.categorical_crossentropy

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        self.weights = self.model.get_weights()


class Conv1DNet(AbstractHasSpotsNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size, passband, **kwargs)

        if not self._from_pickle and self._reinitialize_feed:
            self.train_xs = np.expand_dims(self.train_xs, axis=2)
            self.test_xs = np.expand_dims(self.test_xs, axis=2)

            self.train_ys = to_categorical(self.train_ys, self._n_class)
            self.test_ys = to_categorical(self.test_ys, self._n_class)

        logger.info("creating neural model")
        self.model = tf.keras.Sequential()
        vector_size = 100  # self.train_xs.shape[1]
        self.model.add(layers.Convolution1D(64, 20, activation=nn.tanh, input_shape=(vector_size, 1)))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Convolution1D(32, 10, activation=nn.tanh))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Convolution1D(16, 5, activation=nn.tanh))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation=nn.tanh))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Dense(2, activation=nn.softmax))

        optimizer = optimizers.Adam(lr=self._learning_rate, decay=self._optimizer_decay)
        loss_fn = losses.categorical_crossentropy
        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
        self.weights = self.model.get_weights()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('net', nargs='?', choices=['MlpNet', 'Conv1DNet'], help=f'net choice')
    parser.add_argument('--table', type=str, nargs='?', help='table name to read data', default='synthetic_lc')
    parser.add_argument('--passband', type=str, nargs='?', help='passband', default='Generic.Bessell.V')
    parser.add_argument('--test-size', type=float, nargs='?', help='test size', default=0.2)
    parser.add_argument('--epochs', type=int, nargs='?', help='learning epochs', default=100)
    parser.add_argument('--flux-threshold', type=float, nargs='?', help='flux threshold', default=0.0)
    parser.add_argument('--relative-threshold', type=utils.str2bool, nargs='?',
                        help='relative or absolute threshold', default=True)
    parser.add_argument('--learning-rate', type=float, nargs='?', help='learning rate', default=1e-3)
    parser.add_argument('--optimizer-decay', type=float, nargs='?', help='optimizer decay', default=1e-6)
    parser.add_argument('--load-pickle', type=str, nargs='?', help='path to load pickle file', default=None)
    parser.add_argument('--save-pickle', type=str, nargs='?', help='path to save pickle file', default=None)
    parser.add_argument('--save-history', type=str, nargs='?',
                        help='path to json where fit history will be stored', default=None)
    parser.add_argument('--lr-tuning', type=utils.str2bool, nargs='?', help='run learning rate tunning', default=False)
    parser.add_argument('--home', type=str, nargs='?', help='storage for historical data',
                        default=op.join(op.expanduser("~"), ".elisa"))
    parser.add_argument('--save-model', type=str, nargs='?', help='path to h5 file for model save', default=None)

    args = parser.parse_args()
    getattr(sys.modules[__name__], args.net)
    base.main(args, sys.modules[__name__])
