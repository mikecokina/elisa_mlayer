import numpy as np


DB_CONF = {
    "mysql_user": "root",
    "mysql_pass": "p4ssw0rd",
    "mysql_host": "localhost",
    "mysql_port": 3306,
    "mysql_database": "elisa_mlayer"
}

PHASES = np.linspace(-0.6, 0.6, 100, endpoint=True)
PASSBAND = ["Generic.Bessell.B", "Generic.Bessell.V", "Generic.Bessell.R"]

DATETIME_MASK = '%Y-%m-%dT%H.%M.%S'
