import json

from sqlalchemy import create_engine, Column, Integer, String, Float, Text
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
        params = Column(Text(length=200), nullable=False, name='params')
        data = Column(Text(length=100000), nullable=False, name='data')
        primary_t_eff = Column(Float(), nullable=False, name='primary_t_eff')
        secondary_t_eff = Column(Float(), nullable=False, name='secondary_t_eff')
        primary_surface_potential = Column(Float(), nullable=False, name='primary_surface_potential')
        secondary_surface_potential = Column(Float(), nullable=False, name='secondary_surface_potential')
        primary_mass = Column(Float(), nullable=False, name='primary_mass')
        secondary_mass = Column(Float(), nullable=False, name='secondary_mass')
        period = Column(Float(), nullable=False, name='period')
        inclination = Column(Float(), nullable=False, name='inclination')

    return SyntheticLightCurvesModel


class MySqlIO(object):
    builder_fn = staticmethod(build_synthetic_lightcurve_model)

    def __init__(self, db_conf, table_name):
        self._db_conf = db_conf.copy()

        self.__model_instance = None
        self.__model_declarative_meta = None

        self._table_name = table_name
        self._engine = _engine_pool.get_or_create_engine(db_conf)

    @property
    def _model_instance(self):
        if self.__model_instance is None:
            self._initialise_model()
        return self.__model_instance

    @property
    def _model_declarative_meta(self):
        if self.__model_declarative_meta is None:
            self._initialise_model()
        return self.__model_declarative_meta

    def _initialise_model(self):
        self.__model_declarative_meta = self.builder_fn(table_name=self._table_name)
        self._model_declarative_meta.metadata.create_all(self._engine)
        self.__model_instance = self._model_declarative_meta.metadata.tables.get(self._table_name)

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

    def save(self, data, params, morphology):
        _session = self._get_session()

        if self.__model_instance is None:
            self._initialise_model()

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
            self._model_declarative_meta.inclination.name: float(params["system"]["inclination"])
        }
        new_record = self._model_declarative_meta(**params)
        _session.add(new_record)
        self.finish_session(_session, w=True)

    def get_batch_iter(self, morphology, batch_size):
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
        return _iter


class CsvIO(object):
    pass


def get_mysqlio(db_conf, table_name=conf.DEFAULT_MYSQLIO_TABLE_NAME):
    return MySqlIO(db_conf, table_name=table_name)
