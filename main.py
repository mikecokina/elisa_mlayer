from elisa_mlayer.gen import runner, conf
from elisa_mlayer.config import DB_CONF, PHASES
from elisa_mlayer import sio

PARAMS = {
        "system": {
            "inclination": (80.0, 95.0, 5.0),
            "period": (0.4, 0.5, 0.1)
        },
        "primary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (5000.0, 5500.0, 500.0),
        },
        "secondary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (5000.0, 5500.0, 500.0)
        }
    }


def main():
    _runner = runner.Runner(morphology="over-contact",
                            db_conf=DB_CONF,
                            params=PARAMS,
                            phases=PHASES,
                            threshold=0.01,
                            table_name=conf.DEFAULT_MYSQLIO_TABLE_NAME,
                            io_cls=sio.SyntheticFlatMySqlIO)

    _runner.run()

    # from elisa_mlayer import sio
    # from elisa_mlayer.gen import plot
    # storage = sio.get_mysqlio(DB_CONF, "synthetic_lc")
    # gen = storage.get_batch_iter(morphology="over-contact", batch_size=10, limit=np.inf)
    #
    # plt = plot.Plot()
    # plt.dataset(gen, passband="Generic.Bessell.V")


if __name__ == "__main__":
    main()
