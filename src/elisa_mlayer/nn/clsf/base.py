import pickle
import json
import os
import os.path as op
import numpy as np

from datetime import datetime
from elisa.logger import getLogger
from elisa_mlayer import config

logger = getLogger("nn.clsf.base")


class KerasNet(object):
    def __init__(self, test_size, **kwargs):
        self._test_size = float(test_size)
        self._from_pickle = kwargs.get('pickle', False)
        self._reinitialize_feed = kwargs.get("reinitialize_feed", True)

        self.model = None
        self._feed = None
        self.weights = None
        self.history = None

        if self._reinitialize_feed:
            self.train_xs, self.train_ys, self.test_xs, self.test_ys = None, None, None, None
            if self._from_pickle:
                logger.info("loading feed from pickle")
                self.load_feed(self._from_pickle)
                self._from_pickle = True

        self._learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self._optimizer_decay = float(kwargs.get("optimizer_decay", 1e-6))

    @property
    def model_precission(self):
        val_loss, val_acc = self.model.evaluate(self.test_xs, self.test_ys)
        return {"loss": val_loss, "accuracy": val_acc}

    @property
    def val_loss(self):
        val_loss, _ = self.model.evaluate(self.test_xs, self.test_ys)
        return val_loss

    @property
    def val_acc(self):
        _, val_acc = self.model.evaluate(self.test_xs, self.test_ys)
        return val_acc

    def train(self, epochs):
        """
        :return: history
        """
        self.history = self.model.fit(self.train_xs, self.train_ys, epochs=epochs,
                                      validation_data=(self.test_xs, self.test_ys))
        return self.history

    def save_feed(self, fpath):
        pickle.dump((self.train_xs, self.train_ys, self.test_xs, self.test_ys), open(fpath, "wb"))

    def load_feed(self, fpath):
        self.train_xs, self.train_ys, self.test_xs, self.test_ys = pickle.load(open(fpath, "rb"))

    def save_history(self, fpath):
        if self.history is not None:
            with open(fpath, "w") as f:
                f.write(json.dumps(self.history.history, indent=4))

    def reset_weights(self):
        if self.weights is not None and self.model is not None:
            self.model.set_weights(self.weights)

    def save_weights(self, fpath):
        if self.model is not None:
            base_dir = op.dirname(fpath)
            if not op.isdir(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            fpath = f"{fpath}.h5" if not str(fpath).endswith("h5") else fpath
            self.model.save_weights(fpath)
            return
        raise IOError("No model initialized.")

    def load_weights(self, fpath):
        if self.model is not None and op.isfile(fpath):
            self.model.load_weights(fpath)
            return
        raise IOError("No model initialized or invalid path.")

    def predict(self, xs, argmax=False):
        predictions = self.model.predict(xs)
        if argmax:
            predictions = [np.argmax(pred) for pred in predictions]
        return predictions


class KerasSequenceNet(object):
    def __init__(self):
        pass

    """
    
    classifier = Sequential()
classifier.add(Convolution1D(64, 20, activation='relu', input_shape=(201, 1)))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Convolution1D(32, 10, activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(3, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    """


def main(args, modules):
    if args.net is None:
        raise ValueError("Positional argument is required, choices: `MlpNet`, `Conv1DNet`")

    net = getattr(modules, args.net)
    params = dict(
        table_name=args.table,
        spotty=args.spotty,
        learning_rate=args.learning_rate,
        optimizer_decay=args.optimizer_decay,
        pickle=args.load_pickle or None

    )
    _nn = net(test_size=args.test_size, passband=args.passband, **params)

    if not args.lr_tuning:
        if args.save_pickle is not None:
            _nn.save_feed(args.save_pickle)

        _nn.train(epochs=args.epochs)

        if args.save_model is not None:
            logger.info("saving model")
            _nn.save_weights(args.save_model)

        if args.save_history is not None:
            _nn.save_history(args.save_history)

        logger.info(f'model precision: {_nn.model_precission}')
        logger.info(_nn.model.summary())

    else:
        lr_s = [1e-6, 1e-5, 1e-4, 1e-3, 4e-3, 7e-3, 1e-2, 3e-2, 1e-1]
        loss_history, val_loss_history = [], []
        for learning_rate in lr_s:
            params.update(dict(
                learning_rate=learning_rate,
                optimizer_decay=0.0,
                reinitialize_feed=False
            ))
            _nn.__init__(test_size=args.test_size, passband=args.passband, **params)
            _nn.reset_weights()
            _nn.train(epochs=args.epochs)
            loss_history.append(_nn.history.history["loss"])
            val_loss_history.append(_nn.history.history["val_loss"])

        data = json.dumps({
            "lr_s": lr_s,
            "loss_history": loss_history,
            "val_loss_history": val_loss_history
        }, indent=4)

        logger.info(data)
        if args.home is not None:
            if not op.isdir(args.home):
                os.makedirs(args.home, exist_ok=True)

            now = datetime.now()
            filename = f'{now.strftime(config.DATETIME_MASK)}.json'
            with open(op.join(args.home, filename), "w") as f:
                f.write(data)
