class KerasNet(object):
    def __init__(self, test_size, **kwargs):
        self._test_size = float(test_size)
        self.model = None
        self._feed = None
        self.train_xs, self.train_ys, self.test_xs, self.test_ys = None, None, None, None

        self._learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self._optimizer_decay = float(kwargs.get("optimizer_decay", 1e-6))

    @property
    def model_precission(self):
        val_loss, val_acc = self.model.evaluate(self.test_xs, self.test_ys)
        return {"loss": val_loss, "accuracy": val_acc}

    def train(self, epochs):
        self.model.fit(self.train_xs, self.train_ys, epochs=epochs)


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