import numpy as np
import json

from elisa_mlayer import sio
from elisa_mlayer import config

from matplotlib import pyplot as plt


morphologies = ['over-contact', 'detached']
observed_io = sio.ObservedMySqlIO(db_conf=config.DB_CONF, table_name='observed_lc')


if __name__ == "__main__":
    observed_io._preinit_method()
    session = observed_io._get_session()

    data = session.query(observed_io._model_instance).all()
    observed_io.finish_session(session, w=False)

    for row in data:
        idx = row[0]
        data = json.loads(row[4])
        target = row[7]

        xs = np.arange(0, len(data["flux"]))
        ys = data["flux"]

        plt.plot(xs, ys, label=target)
        plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
        plt.show()
        plt.clf()

