import json
import time
import numpy as np

from numpy import random
from elisa.logger import getPersistentLogger

from elisa_mlayer import config
from elisa_mlayer.io import MySqlIO

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

        return train_samples_xs, train_samples_ys, test_samples_xs, test_samples_ys


class MlpNet(object):
    def __init__(self):
        pass



if __name__ == "__main__":
    feed = Feed(config.DB_CONF, table_name="synthetic_lc")
    feed.get_feed()

