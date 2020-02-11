"""
GUIDE
=====

Recomended parameters
---------------------

detached:

    PARMAS = {
        "system": {
            "inclination": (30.0, 95.0, 5.0),
            # reference: https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/catalogue-of-stellar-parameters-from-the-detached-doublelined-eclipsing-binaries-in-the-milky-way/678AD993366CA4506DD671527906FCBD/core-reader
            "period": (7.5, 40, 2.5) # lower density (lower density is based on information that there is less light curves in such period range)
            "period": (0.4, 7.5, 0.5) # higher density
        },
        "primary": {
            # reference: https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/catalogue-of-stellar-parameters-from-the-detached-doublelined-eclipsing-binaries-in-the-milky-way/678AD993366CA4506DD671527906FCBD/core-reader
            "mass": (0.2, 3.25, 0.25),
            "surface_potential": (2.0, 6.0, 0.25),
            "t_eff": (4000.0, 6500.0, 500.0)
        },
        "secondary": {
            "mass": (0.2, 3.25, 0.25),
            "surface_potential": (2.0, 6.0, 0.25),
            "t_eff": (4000.0, 6500.0, 500.0)
        }
    }


over-contact:

    PARMAS = {
        "system": {
            "inclination": (30.0, 95.0, 5.0),
            "period": (0.2, 0.9, 0.1) # reference: https://academic.oup.com/mnras/article/382/1/393/984693
        },
        "primary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (4000.0, 6500.0, 500.0)
        },
        "secondary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (4000.0, 6500.0, 500.0)
        }
    }


"""

import numpy as np
from elisa_mlayer.gen import cli


DB_CONF = {
        "mysql_user": "root",
        "mysql_pass": "p4ssw0rd",
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "elisa_mlayer"
    }


PHASES = np.linspace(-0.6, 0.6, 100, endpoint=True)
PASSBAND = ["Generic.Bessell.B", "Generic.Bessell.V", "Generic.Bessell.R"]
PARMAS = {
    "system": {
        "inclination": (30.0, 95.0, 5.0),
        "period": (0.2, 0.9, 0.1)
    },
    "primary": {
        "mass": (0.5, 3.25, 0.25),
        "surface_potential": (2.0, 4.0, 0.25),
        "t_eff": (4000.0, 6500.0, 500.0)
    },
    "secondary": {
        "mass": (0.5, 3.25, 0.25),
        "surface_potential": (2.0, 4.0, 0.25),
        "t_eff": (4000.0, 6500.0, 500.0)
    }
}


def main():
    _cli = cli.CLI(morphology="over-contact",
                   db_conf=DB_CONF,
                   params=PARMAS,
                   phases=PHASES,
                   passband=PASSBAND,
                   threshold=0.01)

    _cli.run()


if __name__ == "__main__":
    main()
