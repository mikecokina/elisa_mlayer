def get_albedo(t_eff):
    return 0.5 if t_eff < 6500 else 1.0


def get_gravity_darkening(t_eff):
    return 0.25 if t_eff > 6500 else 0.09


def binary_instance_str():
    return "<class \'elisa.binary_system.system.BinarySystem\'>"


def lc_to_json_serializable(data):
    result = [list(), dict()]
    if not isinstance(data[0], list):
        result[0] = data[0].tolist()
    for band in data[1]:
        if not isinstance(data[1][band], list):
            result[1][band] = data[1][band].tolist()
    return result
