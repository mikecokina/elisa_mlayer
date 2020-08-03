import time
from numpy import random
import numpy as np

from elisa_mlayer import config
from elisa_mlayer.gen import conf
from elisa_mlayer.nn.clsf.base import KerasNet
from elisa_mlayer.sio import SyntheticFlatMySqlIO
from elisa_mlayer.logger import getLogger
from elisa.analytics.binary.params import normalize_value

random.seed(int(time.time()))
logger = getLogger("nn.clsf.morphology")


class Feed(SyntheticFlatMySqlIO):
    def get_feed(self, test_size=0.2, passband='Generic.Bessell.V'):
        logger.info('initializing db session')
        self._preinit_method()
        session = self._get_session()

        logger.info('getting training data')

        data = session.query(self._model_instance) \
            .with_entities(
            self._model_declarative_meta.params,
            self._model_declarative_meta.primary_mass,
            self._model_declarative_meta.secondary_mass,
            getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
        ).all()

        logger.info('shuffling data')
        self.finish_session(session, w=False)
        np.random.shuffle(data)

        logger.info('preparing test and training batches')
        params, m1, m2, xs = zip(*data)


        return None, None, None, None


class AbstractMorphologysNet(KerasNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size=test_size, **kwargs)

        self._passband = str(passband)
        self._table_name = str(kwargs.get("table_name", "synthetic_lc"))

        logger.info("obtaining training data")

        self._feed = Feed(config.DB_CONF, table_name=self._table_name)
        self.train_xs, self.train_ys, self.test_xs, self.test_ys = \
            self._feed.get_feed(test_size=test_size, passband=passband)


class Conv1DNet(AbstractMorphologysNet):
    def __init__(self, test_size, passband='Generic.Bessell.V', **kwargs):
        super().__init__(test_size, passband, **kwargs)


_nn = Conv1DNet(test_size=0.2)
