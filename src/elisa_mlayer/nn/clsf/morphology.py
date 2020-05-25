import argparse
import json
import os
import os.path as op
import sys
import time
import numpy as np
import tensorflow as tf

from datetime import datetime
from keras.utils import to_categorical
from numpy import random

from elisa_mlayer import (
    config,
    utils
)
from elisa_mlayer.gen import conf
from elisa_mlayer.logger import getLogger
from elisa_mlayer.nn.base import layers, losses, nn, optimizers
from elisa_mlayer.nn.clsf.base import KerasNet
from elisa_mlayer.sio import SyntheticFlatMySqlIO

random.seed(int(time.time()))
logger = getLogger("nn.clsf.mlp.morphology")


class Feed(SyntheticFlatMySqlIO):
    def get_feed(self, test_size=0.2, passband='Generic.Bessell.V', spotty=False):

        logger.info('initializing db session')
        self._preinit_method()
        session = self._get_session()

        logger.info('getting training data')
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

        logger.info('shuffling data')
        self.finish_session(session, w=False)
        np.random.shuffle(data)

        logger.info('preparing test and training batches')
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
        super().__init__(test_size=test_size, **kwargs)

        self._n_class = 2
        self._passband = str(passband)
        self._table_name = str(kwargs.get("table_name", "synthetic_lc"))
        self._spotty = kwargs.get("spotty", False)

        if not self._from_pickle and self._reinitialize_feed:
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

        optimizer = optimizers.Adam(lr=self._learning_rate, decay=self._optimizer_decay)
        loss_fn = losses.sparse_categorical_crossentropy

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        self.weights = self.model.get_weights()


class Conv1DNet(AbstractMorphologysNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size, passband, **kwargs)

        if not self._from_pickle and self._reinitialize_feed:
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
    parser.add_argument('--spotty', type=utils.str2bool, nargs='?', help='learning epochs', default=False)
    parser.add_argument('--learning-rate', type=float, nargs='?', help='learning rate', default=1e-3)
    parser.add_argument('--optimizer-decay', type=float, nargs='?', help='optimizer decay', default=1e-6)
    parser.add_argument('--load-pickle', type=str, nargs='?', help='path to load pickle file', default=None)
    parser.add_argument('--save-pickle', type=str, nargs='?', help='path to save pickle file', default=None)
    parser.add_argument('--save-history', type=str, nargs='?',
                        help='path to json where fit history will be stored', default=None)
    parser.add_argument('--lr-tuning', type=utils.str2bool,
                        nargs='?', help='execute learning rate tunning', default=False)
    parser.add_argument('--home', type=str, nargs='?', help='storage for historical data',
                        default=op.join(op.expanduser("~"), ".elisa"))

    args = parser.parse_args()

    if args.net is None:
        raise ValueError("Positional argument is required, choices: `MlpNet`, `Conv1DNet`")
    net = getattr(sys.modules[__name__], args.net)
    params = dict(
        table_name=args.table,
        spotty=args.spotty,
        learning_rate=args.learning_rate,
        optimizer_decay=args.optimizer_decay,
        pickle=args.load_pickle or None

    )
    conv = net(test_size=args.test_size, passband=args.passband, **params)

    if not args.lr_tuning:
        if args.save_pickle is not None:
            conv.save_feed(args.save_pickle)

        conv.train(epochs=args.epochs)

        if args.save_history is not None:
            conv.save_history(args.save_history)

        logger.info(f'model precision: {conv.model_precission}')
        logger.info(conv.model.summary())
    else:
        lr_s = [1e-5, 1e-4, 1e-3, 4e-3, 7e-3, 1e-2, 3e-2, 1e-1]
        loss_history = []
        for learning_rate in lr_s:
            params.update(dict(
                learning_rate=learning_rate,
                optimizer_decay=0.0,
                reinitialize_feed=False
            ))
            conv.__init__(test_size=args.test_size, passband=args.passband, **params)
            conv.reset_weights()
            conv.train(epochs=args.epochs)
            loss_history.append(conv.history.history["loss"])

        data = json.dumps({
            "lr_s": lr_s,
            "loss_history": loss_history
        }, indent=4)

        logger.info(data)
        if args.home is not None:
            if not op.isdir(args.home):
                os.makedirs(args.home, exist_ok=True)

            now = datetime.now()
            filename = f'{now.strftime(conf.DATETIME_MASK)}.json'
            with open(op.join(args.home, filename), "w") as f:
                f.write(data)

        from matplotlib import pyplot as plt

        for hist, learning_rate in zip(loss_history, lr_s):
            plt.plot(np.arange(0, len(hist)), hist, label=f"lr: {learning_rate}")

        plt.legend()
        plt.show()

        plt.xscale("log")
        plt.plot(lr_s, np.array(loss_history)[:, -1])
        plt.show()

    # predictions = mlp.model.predict(mlp.test_xs)
    # for val, pred in zip(mlp.test_ys, predictions):
    #     print(val, np.argmax(pred))
