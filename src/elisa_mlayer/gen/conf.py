import os
import os.path as op

HOME = op.join(op.expanduser("~/"), ".elisa_mlayer")

if not op.isdir(HOME):
    os.mkdir(HOME)

PARAMETERS_ORDERED_LIST = ["primary_mass",
                           "secondary_mass",
                           "primary_t_eff",
                           "secondary_t_eff",
                           "primary_surface_potential",
                           "secondary_surface_potential",
                           "inclination",
                           "period"]


DEFAULT_MYSQLIO_TABLE_NAME = "synthetic_lc"


PASSBAND_TO_COL = {
    "Generic.Bessell.B": "generic_bessell_b",
    "Generic.Bessell.V": "generic_bessell_v",
    "Generic.Bessell.R": "generic_bessell_r"
}

DATETIME_MASK = '%Y-%m-%dT%H.%M.%S'
