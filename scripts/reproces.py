import numpy as np
import pymysql

pymysql.install_as_MySQLdb()
import MySQLdb
import ast
from scipy.interpolate import Akima1DInterpolator
import json
from elisa_mlayer import sio
from elisa_mlayer import config


HOST = "localhost"
USER = "root"
PWD = "p4ssw0rd"
PHASES = np.linspace(-0.6, 0.6, 100, endpoint=True)


class Database(object):
    def __init__(self):
        self.mysql_conn = MySQLdb.connect(host=HOST,  # your host, usually localhost
                                          user=USER,  # your username
                                          passwd=PWD,  # your password
                                          db="lcs")  # name of the data base
        self.c = self.mysql_conn.cursor()

    def query(self, q, *args):
        self.c.execute(q, *args)

    def fetch_all(self):
        return self.c.fetchall()

    def close(self):
        self.mysql_conn.close()


db = Database()
q = f'SELECT * FROM observation_data'
db.query(q)
dataset = db.fetch_all()
db.close()


def full_phase_to_partial(phase):
    return np.mod(phase, 1.0)


m_map = {
    "over-contact": "over-contact",
    "overcontact": "over-contact",
    "detached": "detached",
    "semi-detached": "detached",
    "semidetached": "detached"
}


for row in dataset:
    _, object_label, morphology, period, epoch, origin_lightcurve, processed_lightcurve, note = row

    curve = ast.literal_eval(processed_lightcurve)
    xs, ys = curve["phase"], curve["flux"]

    if origin_lightcurve is not None:
        try:
            origin_lightcurve = ast.literal_eval(origin_lightcurve)
        except:
            origin_lightcurve = origin_lightcurve.replace("{HJD}", "[]")

            try:
                origin_lightcurve = ast.literal_eval(origin_lightcurve)
            except:
                origin_lightcurve = "{" + origin_lightcurve
                origin_lightcurve = ast.literal_eval(origin_lightcurve)
    else:
        origin_lightcurve = {"flux": [], "phase": []}

    if 'magnitude' in origin_lightcurve:
        ys = -np.array(ys)
        ys = np.power(10, (10 - ys) / 2.5)
        ys = ys / max(ys)
        ys = ys.tolist()

    interpolator = Akima1DInterpolator(xs, ys)

    p_xs = full_phase_to_partial(PHASES)
    p_ys = interpolator(p_xs)

    data = {
        "flux": p_ys.tolist(),
        "phase": PHASES.tolist()
    }

    metadata = {"morphology": morphology}
    if note is not None:
        metadata["note"] = str(note)
    morphology = m_map[morphology]
    epoch = float(epoch)
    period = float(period)
    data = json.dumps(data)
    origin = json.dumps(origin_lightcurve)
    params = "{}"
    target = str(object_label)
    passband = ""
    metadata = json.dumps(metadata)

    # iterable_data = [(morphology, passband, params, data, origin, period, target, epoch, metadata)]
    # storage = sio.ObservedMySqlIO(config.DB_CONF, table_name="observed_lc")
    # storage.save(iterable_data)
