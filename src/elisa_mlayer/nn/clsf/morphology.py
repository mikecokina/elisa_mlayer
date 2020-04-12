import argparse
import time
import numpy as np
import tensorflow as tf
import sys
import json

from numpy import random
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from elisa_mlayer import (
    config,
    utils
)
from elisa_mlayer.gen import conf
from elisa_mlayer.logger import getLogger
from elisa_mlayer.sio import SyntheticFlatMySqlIO
from elisa_mlayer.nn.base import layers, losses, nn, optimizers
from elisa_mlayer.nn.clsf.base import KerasNet

random.seed(int(time.time()))
logger = getLogger("nn.clsf.mlp.morphology")


class Feed(SyntheticFlatMySqlIO):
    def get_feed(self, test_size=0.2, passband='Generic.Bessell.V', spotty=None):
        self._preinit_method()
        session = self._get_session()

        if isinstance(spotty, bool):
            data = session.query(self._model_instance) \
                .filter(self._model_declarative_meta.spotty == spotty) \
                .with_entities(
                self._model_declarative_meta.morphology,
                getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
            ).all()
        else:
            data = session.query(self._model_instance) \
                .with_entities(
                self._model_declarative_meta.morphology,
                getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
            ).all()

        self.finish_session(session, w=False)
        np.random.shuffle(data)

        ys, xs = zip(*data)
        ys, xs = np.array(ys), np.array([json.loads(row) for row in xs])
        xs = np.divide(xs, xs.max(axis=1)[:, None])
        is_detached = ys == 'detached'
        all_entities = ys.shape[0]
        ys = np.zeros(all_entities, dtype=bool)
        ys[is_detached] = True

        test_samples_xs, test_samples_ys = xs[:int(all_entities * test_size)], ys[:int(all_entities * test_size)]
        train_samples_xs, train_samples_ys = xs[int(all_entities * test_size):], ys[int(all_entities * test_size):]

        train_samples_ys = self.parse_y_feed(train_samples_ys)
        test_samples_ys = self.parse_y_feed(test_samples_ys)

        return utils.convert_to_numpy_array((train_samples_xs, train_samples_ys, test_samples_xs, test_samples_ys))

    @classmethod
    def parse_y_feed(cls, y_data):
        return np.array(y_data, dtype=int)


class AbstractMorphologysNet(KerasNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size=test_size)

        self._n_class = (0, 1)
        self._passband = str(passband)
        self._table_name = str(kwargs.get("table_name", "synthetic_lc"))
        self._spotty = str(kwargs.get("spotty", None))

        logger.info("obtaining training data")
        self._feed = Feed(config.DB_CONF, table_name=self._table_name)
        self.train_xs, self.train_ys, self.test_xs, self.test_ys = \
            self._feed.get_feed(test_size=test_size, passband=passband, spotty=self._spotty)


class MlpNet(AbstractMorphologysNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size, passband, **kwargs)

        logger.info("creating neural model")
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(128, activation=nn.relu, input_shape=self.train_xs.shape[1:]))
        self.model.add(layers.Dense(256, activation=nn.relu))
        self.model.add(layers.Dense(2, activation=nn.softmax))

        optimizer = optimizers.Adam(lr=0.001, decay=1e-6)
        loss_fn = losses.SparseCategoricalCrossentropy()

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


class Conv1DNet(AbstractMorphologysNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size, passband, **kwargs)

        self.train_xs = np.expand_dims(self.train_xs, axis=2)
        self.test_xs = np.expand_dims(self.test_xs, axis=2)

        self.train_ys = to_categorical(self.train_ys, self._n_class)
        self.test_ys = to_categorical(self.test_ys, self._n_class)

        logger.info("creating neural model")
        self.model = tf.keras.Sequential()
        self.model.add(layers.Convolution1D(64, 20, activation=nn.relu, input_shape=(self.train_xs.shape[1], 1)))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Convolution1D(32, 10, activation=nn.relu))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation=nn.relu))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Dense(2, activation=nn.softmax))

        optimizer = optimizers.Adam(lr=1e-3, decay=1e-6)
        loss_fn = losses.CategoricalCrossentropy()
        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('net', nargs='?', choices=['MlpNet', 'Conv1DNet'], help=f'net choice')
    parser.add_argument('--table', type=str, nargs='?', help='table name to read data', default='synthetic_lc')
    parser.add_argument('--passband', type=str, nargs='?', help='passband', default='Generic.Bessell.V')
    parser.add_argument('--test-size', type=float, nargs='?', help='test size', default=0.2)
    parser.add_argument('--epochs', type=int, nargs='?', help='learning epochs', default=100)
    parser.add_argument('--spotty', type=utils.str2bool, nargs='?', help='learning epochs', default=None)
    args = parser.parse_args()

    if args.net is None:
        raise ValueError("Positional argument is required, choices: `MlpNet`, `Conv1DNet`")
    net = getattr(sys.modules[__name__], args.net)
    conv = net(test_size=args.test_size, passband=args.passband, table_name=args.table, spotty=args.spotty)
    conv.train(epochs=args.epochs)
    print(conv.model_precission)

    # predictions = mlp.model.predict(mlp.test_xs)
    # for val, pred in zip(mlp.test_ys, predictions):
    #     print(val, np.argmax(pred))
