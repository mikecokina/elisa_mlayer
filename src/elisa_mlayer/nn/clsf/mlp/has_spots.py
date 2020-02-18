import json
import time
import numpy as np
import tensorflow as tf

from numpy import random
from elisa.logger import getPersistentLogger

from elisa_mlayer import (
    config,
    utils
)
from elisa_mlayer.io import MySqlIO
from elisa_mlayer.nn.base import layers, losses, nn, optimizers

random.seed(int(time.time()))
logger = getPersistentLogger("nn.clsf.mlp.has_spots")


class Feed(MySqlIO):
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


class MlpNet(object):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        self._passband = str(passband)
        self._test_size = float(test_size)
        self._table_name = str(kwargs.get("table_name", "synthetic_lc"))

        logger.info("obtaining training data")
        self._feed = Feed(config.DB_CONF, table_name=self._table_name)
        self.train_xs, self.train_ys, self.test_xs, self.test_ys = self._feed.get_feed()

        logger.info("creating neural model")

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(128, activation=nn.relu, input_shape=self.train_xs.shape[1:]))
        self.model.add(layers.Dense(64, activation=nn.relu))
        self.model.add(layers.Dense(2, activation=nn.softmax))

        optimizer = optimizers.Adam(lr=0.001, decay=1e-6)
        loss_fn = losses.SparseCategoricalCrossentropy()

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        print(self.train_xs)

    def train(self, epochs):
        self.model.fit(self.train_xs, self.train_ys, epochs=epochs)

    @property
    def model_precission(self):
        val_loss, val_acc = self.model.evaluate(self.test_xs, self.test_ys)
        return {"loss": val_loss, "accuracy": val_acc}


if __name__ == "__main__":
    mlp = MlpNet(test_size=0.2)
    mlp.train(epochs=1000)

    predictions = mlp.model.predict(mlp.test_xs)
    print([np.argmax(val) for val in predictions])


