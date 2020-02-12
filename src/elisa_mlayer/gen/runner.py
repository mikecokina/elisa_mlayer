from elisa_mlayer.gen import io, generate
from elisa.logger import getPersistentLogger

logger = getPersistentLogger('cli')


class Runner(object):
    def __init__(self, morphology, db_conf, params, phases, passband, threshold):
        self._morphology = morphology
        self._phases = phases
        self._threshold = threshold
        self._passband = passband
        self._db_conf = db_conf
        self._params = params
        self._storage = io.get_mysqlio(self._db_conf)
        self._generator = self._get_generator()

    def _get_generator(self):
        if self._morphology in ["detached"]:
            return generate.DetachedLCGenerator(phases=self._phases,
                                                passband=self._passband,
                                                threshold=self._threshold,
                                                **self._params)
        elif self._morphology in ["over-contact"]:
            return generate.OverContactLCGenerator(phases=self._phases,
                                                   passband=self._passband,
                                                   threshold=self._threshold,
                                                   **self._params)
        else:
            raise ValueError("Invalid morphology. Allowed [detached, over-contact]")

    def run(self):
        logger.info("Starting generator")
        for _data in self._generator:
            self._storage.save(*_data)
