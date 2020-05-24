import pickle
import json
from elisa.logger import getLogger

logger = getLogger("nn.clsf.base")


class KerasNet(object):
    def __init__(self, test_size, **kwargs):
        self._test_size = float(test_size)
        self._from_pickle = kwargs.get('pickle', False)
        self.model = None
        self._feed = None
        self.weights = None
        self.history = None

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
    def loss(self):
        val_loss, _ = self.model.evaluate(self.test_xs, self.test_ys)
        return val_loss

    @property
    def acc(self):
        _, val_acc = self.model.evaluate(self.test_xs, self.test_ys)
        return val_acc
    
    accuracy = acc

    def train(self, epochs):
        """
        :return: history
        """
        self.history = self.model.fit(self.train_xs, self.train_ys, epochs=epochs)
        return self.history

    def save_feed(self, fpath):
        pickle.dump((self.train_xs, self.train_ys, self.test_xs, self.test_ys), open(fpath, "wb"))

    def load_feed(self, fpath):
        self.train_xs, self.train_ys, self.test_xs, self.test_ys = pickle.load(open(fpath, "rb"))

    def reset_weights(self):
        if self.weights is not None and self.model is not None:
            self.model.set_weight(self.weights)

    def save_history(self, fpath):
        if self.history is not None:
            with open(fpath, "w") as f:
                f.write(json.dumps(self.history.history, indent=4))


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
