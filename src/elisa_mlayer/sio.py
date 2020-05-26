import json
import numpy as np

from abc import abstractmethod
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from elisa_mlayer.gen import conf, utils


class MySqlEnginePool:

    def __init__(self):
        self._engines = {}

    def get_or_create_engine(self, history_conf):
        conn_str = self._get_conn_str(history_conf)
        try:
            return self._engines[conn_str]
        except KeyError:
            engine = self._create_engine(conn_str)
            self._engines[conn_str] = engine
            return engine

    @staticmethod
    def _get_conn_str(db_conf):
        conn_str = 'mysql+pymysql://{}:{}@{}/{}?host={}?port={}'.format(
            db_conf.get("mysql_user"),
            db_conf.get("mysql_pass"),
            db_conf.get("mysql_host"),
            db_conf.get("mysql_database"),
            db_conf.get("mysql_host"),
            db_conf.get("mysql_port")
        )
        return conn_str

    @classmethod
    def _create_engine(cls, conn_str):
        return create_engine(
            conn_str,
            pool_size=1,
            pool_recycle=3600,
        )

    def dispose(self):
        for engine in self._engines.values():
            engine.dispose()


_engine_pool = MySqlEnginePool()


def build_synthetic_lightcurve_model(table_name):
    class SyntheticLightCurvesModel(declarative_base()):
        __tablename__ = table_name

        id = Column(Integer, primary_key=True, name='id', autoincrement=True)
        morphology = Column(String(length=30), nullable=False, name='morphology')
        params = Column(Text(length=5000), nullable=False, name='params')
        data = Column(Text(length=100000), nullable=False, name='data')
        primary_t_eff = Column(Float(), nullable=False, name='primary_t_eff')
        secondary_t_eff = Column(Float(), nullable=False, name='secondary_t_eff')
        primary_surface_potential = Column(Float(), nullable=False, name='primary_surface_potential')
        secondary_surface_potential = Column(Float(), nullable=False, name='secondary_surface_potential')
        primary_mass = Column(Float(), nullable=False, name='primary_mass')
        secondary_mass = Column(Float(), nullable=False, name='secondary_mass')
        period = Column(Float(), nullable=False, name='period')
        inclination = Column(Float(), nullable=False, name='inclination')
        spotty = Column(Boolean(), nullable=False, name='spotty')

    return SyntheticLightCurvesModel


def build_synthetic_lightcurve_model_flat(table_name):
    class SyntheticLightCurvesModelFlat(declarative_base()):
        __tablename__ = table_name

        id = Column(Integer, primary_key=True, name='id', autoincrement=True)
        morphology = Column(String(length=30), nullable=False, name='morphology')
        params = Column(Text(length=5000), nullable=False, name='params')
        phases = Column(Text(length=100000), nullable=False, name='phases')
        generic_bessell_b = Column(Text(length=100000), nullable=False, name='generic_bessell_b')
        generic_bessell_v = Column(Text(length=100000), nullable=False, name='generic_bessell_v')
        generic_bessell_r = Column(Text(length=100000), nullable=False, name='generic_bessell_r')
        primary_t_eff = Column(Float(), nullable=False, name='primary_t_eff')
        secondary_t_eff = Column(Float(), nullable=False, name='secondary_t_eff')
        primary_surface_potential = Column(Float(), nullable=False, name='primary_surface_potential')
        secondary_surface_potential = Column(Float(), nullable=False, name='secondary_surface_potential')
        primary_mass = Column(Float(), nullable=False, name='primary_mass')
        secondary_mass = Column(Float(), nullable=False, name='secondary_mass')
        period = Column(Float(), nullable=False, name='period')
        inclination = Column(Float(), nullable=False, name='inclination')
        spotty = Column(Boolean(), nullable=False, name='spotty')

    return SyntheticLightCurvesModelFlat


def build_observed_lightcurve_model(table_name):
    class ObservedLightCurvesModel(declarative_base()):
        __tablename__ = table_name

        id = Column(Integer, primary_key=True, name='id', autoincrement=True)
        morphology = Column(String(length=30), nullable=False, name='morphology')
        passband = Column(String(length=64), nullable=True, name='passband')
        params = Column(Text(length=5000), nullable=True, name='params')
        data = Column(Text(length=100000), nullable=False, name='data')
        origin = Column(Text(length=2000000), nullable=True, name='origin')
        period = Column(Float(), nullable=False, name='period')
        target = Column(String(length=128), nullable=True, name='target')
        epoch = Column(Float(), nullable=True, name='epoch')
        meta = Column(Text(length=10000), nullable=True, name='meta')

    return ObservedLightCurvesModel


class AbstractMySqlIO(object):
    @staticmethod
    @abstractmethod
    def builder_fn(*args, **kwargs):
        pass

    def __init__(self, db_conf, table_name):
        self._db_conf = db_conf.copy()

        self._table_name = table_name
        self._engine = _engine_pool.get_or_create_engine(db_conf)

        self._model_instance = None
        self._model_declarative_meta = None

    @property
    def model_instance(self):
        if self._model_instance is None:
            self._initialise_model()
        return self._model_instance

    @property
    def model_declarative_meta(self):
        if self._model_declarative_meta is None:
            self._initialise_model()
        return self._model_declarative_meta

    def _initialise_model(self):
        self._model_declarative_meta = self.builder_fn(table_name=self._table_name)
        self._model_declarative_meta.metadata.create_all(self._engine)
        self._model_instance = self._model_declarative_meta.metadata.tables.get(self._table_name)

    def _get_session(self):
        # Create sample session
        __session = sessionmaker()
        __session.configure(bind=self._engine)
        return __session()

    @staticmethod
    def finish_session(session, w=False):
        if w:
            session.flush()
            session.commit()
        session.close()

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    def _preinit_method(self):
        if self._model_instance is None:
            self._initialise_model()


class SyntheticMySqlIO(AbstractMySqlIO):
    builder_fn = staticmethod(build_synthetic_lightcurve_model)

    def __init__(self, db_conf, table_name):
        super().__init__(db_conf, table_name)

    def save(self, iterable):
        self._preinit_method()
        _session = self._get_session()

        for row in iterable:
            data, params, morphology, spotty = row

            params = {
                self._model_declarative_meta.morphology.name: morphology,
                self._model_declarative_meta.params.name: json.dumps(params),
                self._model_declarative_meta.data.name: json.dumps(utils.lc_to_json_serializable(data)),
                self._model_declarative_meta.primary_t_eff.name: float(params["primary"]["t_eff"]),
                self._model_declarative_meta.secondary_t_eff.name: float(params["secondary"]["t_eff"]),
                self._model_declarative_meta.primary_surface_potential.name: float(params["primary"]["surface_potential"]),
                self._model_declarative_meta.secondary_surface_potential.name: float(
                    params["secondary"]["surface_potential"]),
                self._model_declarative_meta.primary_mass.name: float(params["primary"]["mass"]),
                self._model_declarative_meta.secondary_mass.name: float(params["secondary"]["mass"]),
                self._model_declarative_meta.period.name: float(params["system"]["period"]),
                self._model_declarative_meta.inclination.name: float(params["system"]["inclination"]),
                self._model_declarative_meta.spotty.name: spotty
            }
            new_record = self._model_declarative_meta(**params)
            _session.add(new_record)

        self.finish_session(_session, w=True)

    def get_batch_iter(self, morphology, batch_size, limit=np.inf):
        self._preinit_method()
        _session = self._get_session()

        def _iter():
            loop_index = 0

            while True:

                result = _session.query(self._model_instance) \
                    .filter(self._model_declarative_meta.morphology == morphology) \
                    .offset(loop_index * batch_size) \
                    .limit(batch_size) \
                    .all()

                loop_index += 1
                if len(result) == 0:
                    break

                yield result

                if loop_index > limit:
                    break

        return _iter


class ObservedMySqlIO(AbstractMySqlIO):
    builder_fn = staticmethod(build_observed_lightcurve_model)

    def __init__(self, db_conf, table_name):
        self.__model_instance = None
        self.__model_declarative_meta = None
        super().__init__(db_conf, table_name)

    def save(self, iterable):
        self._preinit_method()
        _session = self._get_session()

        for row in iterable:
            morphology, passband, params, data, origin, period, target, epoch, meta = row

            params = {
                self._model_declarative_meta.morphology.name: morphology,
                self._model_declarative_meta.passband.name: passband,
                self._model_declarative_meta.params.name: params,
                self._model_declarative_meta.data.name: data,
                self._model_declarative_meta.origin.name: origin,
                self._model_declarative_meta.period.name: period,
                self._model_declarative_meta.target.name: target,
                self._model_declarative_meta.epoch.name: epoch,
                self._model_declarative_meta.meta.name: meta
            }
            new_record = self._model_declarative_meta(**params)
            _session.add(new_record)

        self.finish_session(_session, w=True)

    def get_predictor_iter(self, batch_size, morphology=None, limit=np.inf):
        self._preinit_method()
        _session = self._get_session()

        def _iter():
            loop_index = 0

            while True:

                if morphology is not None:
                    result = _session.query(self._model_instance) \
                        .filter(self._model_declarative_meta.morphology == morphology) \
                        .with_entities(self._model_declarative_meta.morphology,
                                       self._model_declarative_meta.data) \
                        .offset(loop_index * batch_size) \
                        .limit(batch_size) \
                        .all()
                else:
                    result = _session.query(self._model_instance) \
                        .with_entities(self._model_declarative_meta.morphology,
                                       self._model_declarative_meta.data) \
                        .offset(loop_index * batch_size) \
                        .limit(batch_size) \
                        .all()

                loop_index += 1
                if len(result) == 0:
                    break

                ys, xs = zip(*result)
                ys, xs = np.array(ys), np.array([json.loads(row)["flux"] for row in xs])
                is_detached = ys == 'detached'
                all_entities = ys.shape[0]
                ys = np.zeros(all_entities, dtype=bool)
                ys[is_detached] = True

                yield xs, ys

                if loop_index > limit:
                    break

        return _iter


class SyntheticFlatMySqlIO(AbstractMySqlIO):
    builder_fn = staticmethod(build_synthetic_lightcurve_model_flat)

    def __init__(self, db_conf, table_name):
        super().__init__(db_conf, table_name)

    def save(self, iterable):
        """
        :param iterable: List; [(data, params, morphology, spotty), ...]
        """
        self._preinit_method()
        _session = self._get_session()

        for row in iterable:
            data, params, morphology, spotty = row

            data = utils.lc_to_json_serializable(data)

            params = {
                self._model_declarative_meta.morphology.name: morphology,
                self._model_declarative_meta.params.name: json.dumps(params),
                self._model_declarative_meta.phases.name: json.dumps(data[0]),
                self._model_declarative_meta.generic_bessell_b.name: json.dumps(data[1]['Generic.Bessell.B']),
                self._model_declarative_meta.generic_bessell_v.name: json.dumps(data[1]['Generic.Bessell.V']),
                self._model_declarative_meta.generic_bessell_r.name: json.dumps(data[1]['Generic.Bessell.R']),
                self._model_declarative_meta.primary_t_eff.name: float(params["primary"]["t_eff"]),
                self._model_declarative_meta.secondary_t_eff.name: float(params["secondary"]["t_eff"]),
                self._model_declarative_meta.primary_surface_potential.name:
                    float(params["primary"]["surface_potential"]),
                self._model_declarative_meta.secondary_surface_potential.name:
                    float(params["secondary"]["surface_potential"]),
                self._model_declarative_meta.primary_mass.name: float(params["primary"]["mass"]),
                self._model_declarative_meta.secondary_mass.name: float(params["secondary"]["mass"]),
                self._model_declarative_meta.period.name: float(params["system"]["period"]),
                self._model_declarative_meta.inclination.name: float(params["system"]["inclination"]),
                self._model_declarative_meta.spotty.name: spotty
            }
            new_record = self._model_declarative_meta(**params)
            _session.add(new_record)

        self.finish_session(_session, w=True)

    def spotty_batch_itter(self, batch_size, passband, morphology=None, limit=np.inf):
        self._preinit_method()
        _session = self._get_session()

        def _iter():
            loop_index = 0

            while True:

                if morphology:
                    result = _session.query(self._model_instance)\
                        .filter(self._model_declarative_meta.morphology == morphology)\
                        .offset(loop_index * batch_size)\
                        .limit(batch_size)\
                        .with_entities(
                        getattr(self._model_declarative_meta, "spotty"),
                        getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
                    ).all()
                else:
                    result = _session.query(self._model_instance) \
                        .offset(loop_index * batch_size) \
                        .limit(batch_size) \
                        .with_entities(
                        getattr(self._model_declarative_meta, "spotty"),
                        getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
                    ).all()

                loop_index += 1
                if len(result) == 0:
                    break

                yield result

                if loop_index > limit:
                    break

        return _iter

    def morphology_batch_itter(self, batch_size, passband, spotty=None, limit=np.inf):
        self._preinit_method()
        _session = self._get_session()

        def _iter():
            loop_index = 0

            while True:

                if isinstance(spotty, bool):

                    result = _session.query(self._model_instance)\
                        .filter(self._model_declarative_meta.spotty == spotty)\
                        .offset(loop_index * batch_size)\
                        .limit(batch_size)\
                        .with_entities(
                        getattr(self._model_declarative_meta, "morphology"),
                        getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
                    ).all()
                else:
                    result = _session.query(self._model_instance) \
                        .offset(loop_index * batch_size) \
                        .limit(batch_size) \
                        .with_entities(
                        getattr(self._model_declarative_meta, "morphology"),
                        getattr(self._model_declarative_meta, conf.PASSBAND_TO_COL[passband])
                    ).all()

                loop_index += 1
                if len(result) == 0:
                    break

                yield result

                if loop_index > limit:
                    break

        return _iter


class CsvIO(object):
    pass


def get_mysqlio(db_conf, table_name=conf.DEFAULT_MYSQLIO_TABLE_NAME, io_cls=SyntheticMySqlIO):
    return io_cls(db_conf, table_name=table_name)
