import numpy as np
import json

from elisa_mlayer import sio
from elisa_mlayer import config


morphologies = ['over-contact', 'detached']
std_mysql_io = sio.get_mysqlio(config.DB_CONF, io_cls=sio.SyntheticMySqlIO)
extd_mysql_io = sio.get_mysqlio(config.DB_CONF, table_name="synthetic_lc_col", io_cls=sio.SyntheticExtendedMySqlIO)


if __name__ == "__main__":
    for morphology in morphologies:
        std_gen = std_mysql_io.get_batch_iter(morphology, batch_size=1, limit=np.inf)

        for data in std_gen():
            kwargs = {
                "spotty": data[-1].spotty,
                "data": json.loads(data[-1].data),
                "params": json.loads(data[-1].params),
                "morphology": data[-1].morphology

            }
            extd_mysql_io.save(**kwargs)

    # for morphology in morphologies:
    #     ext_gen = extd_mysql_io.spotty_batch_itter(passband="Generic.Bessell.B", batch_size=1, limit=np.inf)
    #
    #     for data in ext_gen():
    #         print()
