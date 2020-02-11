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
