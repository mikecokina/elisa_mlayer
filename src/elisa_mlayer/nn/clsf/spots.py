import json
import time
import numpy as np
import tensorflow as tf

from numpy import random
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from elisa_mlayer import (
    config,
    utils
)

from elisa_mlayer.logger import getLogger
from elisa_mlayer.io import SyntheticMySqlIO
from elisa_mlayer.nn.base import layers, losses, nn, optimizers
from elisa_mlayer.nn.clsf.base import KerasNet

random.seed(int(time.time()))
logger = getLogger("nn.clsf.mlp.has_spots")


class Feed(SyntheticMySqlIO):
    def get_feed(self, test_size=0.2, passband='Generic.Bessell.V'):
        session = self._get_session()
        data = session.query(self._model_instance).with_entities(self._model_declarative_meta.spotty,
                                                                 self._model_declarative_meta.data).all()
        np.random.shuffle(data)
        self.finish_session(session, w=False)

        ys, xs = list(zip(*((record[0], np.array(json.loads(record[1])[1][passband])) for record in data)))
        xs = [record / record[70] for record in xs]

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
        super().__init__(test_size=test_size)

        self._passband = str(passband)
        self._table_name = str(kwargs.get("table_name", "synthetic_lc"))

        logger.info("obtaining training data")
        self._feed = Feed(config.DB_CONF, table_name=self._table_name)
        self.train_xs, self.train_ys, self.test_xs, self.test_ys = self._feed.get_feed()


class MlpNet(AbstractHasSpotsNet):
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


class Conv1DNet(AbstractHasSpotsNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size, passband, **kwargs)

        self.train_xs = np.expand_dims(self.train_xs, axis=2)
        self.test_xs = np.expand_dims(self.test_xs, axis=2)

        self.train_ys = to_categorical(self.train_ys)
        self.test_ys = to_categorical(self.test_ys)

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
    # mlp = MlpNet(test_size=0.2)
    # mlp.train(epochs=100)
    # print(mlp.model_precission)

    conv = Conv1DNet(test_size=0.2)
    conv.train(epochs=100)
    print(conv.model_precission)

    # predictions = mlp.model.predict(mlp.test_xs)
    # for val, pred in zip(mlp.test_ys, predictions):
    #     print(val, np.argmax(pred))
