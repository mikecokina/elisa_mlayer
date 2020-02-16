import numpy as np
import itertools
import time

from abc import abstractmethod
from numpy import random

from elisa.base.error import MorphologyError, LimbDarkeningError
from elisa.binary_system.system import BinarySystem
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.observer.observer import Observer
from elisa_mlayer.gen import conf, utils
from elisa.logger import getPersistentLogger
from elisa.const import SOLAR_RADIUS

random.seed(int(time.time()))
logger = getPersistentLogger("gen.generate")


class StarDataBoundaries(object):
    def __init__(self, mass, surface_potential, t_eff, spots=None):
        self.mass = mass
        self.surface_potential = surface_potential
        self.t_eff = t_eff
        self.spots = spots

    @classmethod
    def from_json(cls, data):
        return cls(**data)


class SystemDataBoundaries(object):
    def __init__(self, inclination, period):
        self.inclination = inclination
        self.period = period

    @classmethod
    def from_json(cls, data):
        return cls(**data)


class LCGenerator(object):
    MORPHOLOGY = ''

    def __init__(self, phases, passband, threshold, **kwargs):
        self._total = 0
        self._valid_curves = 0
        self.phases = phases

        self.observer = Observer(passband, system=None)
        setattr(self.observer, "_system_cls", utils.binary_instance_str())
        self.system = SystemDataBoundaries.from_json(kwargs.get("system"))
        self.primary = StarDataBoundaries.from_json(kwargs.get("primary"))
        self.secondary = StarDataBoundaries.from_json(kwargs.get("secondary"))
        self.threshold = threshold

        primary_mass = np.arange(*self.primary.mass)
        secondary_mass = np.arange(*self.secondary.mass)
        primary_t_eff = np.arange(*self.primary.t_eff)
        secondary_t_eff = np.arange(*self.secondary.t_eff)
        primary_surface_potential = np.arange(*self.primary.surface_potential)
        secondary_surface_potential = np.arange(*self.secondary.surface_potential)

        inclination = np.arange(*self.system.inclination)
        period = np.arange(*self.system.period)

        self.parameters = itertools.product(*(primary_mass, secondary_mass, primary_t_eff, secondary_t_eff,
                                            primary_surface_potential, secondary_surface_potential,
                                            inclination, period))

        self.spotty_system = True if self.primary.spots or self.secondary.spots else False

    @staticmethod
    def params_to_dict(params):
        return {k: v for k, v in zip(conf.PARAMETERS_ORDERED_LIST, params)}

    @staticmethod
    def params_to_elisa_json(**kwargs):
        return {
            "system": {
                "inclination": kwargs["inclination"],
                "period": kwargs["period"],
                "argument_of_periastron": 90.0,
                "gamma": 0.0,
                "eccentricity": 0.0,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0
            },
            "primary": {
                "mass": kwargs["primary_mass"],
                "surface_potential": kwargs["primary_surface_potential"],
                "synchronicity": 1.0,
                "t_eff": kwargs["primary_t_eff"],
                "gravity_darkening": utils.get_gravity_darkening(kwargs["primary_t_eff"]),
                "discretization_factor": 10,
                "albedo": utils.get_albedo(kwargs["secondary_t_eff"]),
                "metallicity": 0.0,
                "spots": kwargs.get("primary_spots")
            },
            "secondary": {
                "mass": kwargs["secondary_mass"],
                "surface_potential": kwargs["secondary_surface_potential"],
                "synchronicity": 1.0,
                "t_eff": kwargs["secondary_t_eff"],
                "gravity_darkening": utils.get_gravity_darkening(kwargs["secondary_t_eff"]),
                "albedo": utils.get_albedo(kwargs["secondary_t_eff"]),
                "metallicity": 0.0,
                "spots": kwargs.get("secondary_spots")
            }
        }

    def add_spots(self, params):
        for component in BINARY_COUNTERPARTS:
            if getattr(getattr(self, str(component)), "spots"):
                params[f"{component}_spots"] = [
                    {
                        "longitude": random.uniform(*spot_deffiniton["longitude"]),
                        "latitude": random.uniform(*spot_deffiniton["latitude"]),
                        "angular_radius": random.uniform(*spot_deffiniton["angular_radius"]),
                        "temperature_factor": random.uniform(*spot_deffiniton["temperature_factor"])
                    } for spot_deffiniton in getattr(getattr(self, str(component)), "spots")]
        return params

    def __iter__(self):
        for params in self.parameters:
            self._total += 1
            params = self.params_to_dict(params)
            params = self.add_spots(params)
            params = self.params_to_elisa_json(**params)

            try:
                bs = BinarySystem.from_json(params, _verify=False, _kind_of="std")
            except MorphologyError:
                # logger.info(f"hit MorphologyError, continue")
                continue

            if self.kick_by_potential(params["primary"]["surface_potential"], bs):
                # logger.info(f"hit invalid surface potential, max allowed (critical) "
                #             f"{bs.primary.critical_surface_potential}, given: {params['primary']['surface_potential']}")
                continue

            if self.kick_by_radius(bs):
                # logger.info(f"hit invalid radius out of <0.2 - 35> Solar radii, continue")
                continue

            if self.kick_by_eclipse(bs):
                # logger.info(f"hit no-eclipse constellation, continue")
                continue

            if self.eval_morphology_skip(bs):
                # logger.info(f"hit {bs.morphology} system, {self.MORPHOLOGY} expeted, continue")
                continue

            reslc = None
            try:
                self._valid_curves += 1
                logger.info(f"generator finished, hit {self._valid_curves} light curves {self._valid_curves}/{self._total}")
                continue
                reslc = self.generate_lc(bs)
            except LimbDarkeningError:
                logger.info(f"hit LimbDarkeningError, continue")
                continue

            if self.kick_by_threshold(reslc):
                logger.info(f"hit threshold {self.threshold}, continue")
                continue

            self._valid_curves += 1
            logger.info(f"evaluated {self._valid_curves}th light cruve")
            yield reslc, params, bs.morphology, self.spotty_system

        logger.info(f"generator finished, hit {self._valid_curves} light curves")

    def generate_lc(self, bs):
        setattr(self.observer, "_system", bs)
        lc = self.observer.observe.lc(phases=self.phases)
        return lc

    @abstractmethod
    def eval_morphology_skip(self, bs):
        pass

    def kick_by_threshold(self, data):
        norm = [data[1][band] / data[1][band].max() for band in data[1]]
        diff = [data.max() - data.min() for data in norm]
        return True if np.any(np.array(diff) < self.threshold) else False

    @staticmethod
    def kick_by_eclipse(bs):
        return True if ((np.pi / 2.0) - np.arcsin(bs.primary.polar_radius + bs.secondary.polar_radius)) \
                       > bs.inclination else False

    @staticmethod
    def kick_by_potential(surface_potential, bs):
        return False

    @staticmethod
    def kick_by_radius(bs):
        return False


class DetachedLCGenerator(LCGenerator):
    MORPHOLOGY = "detached"

    def eval_morphology_skip(self, bs):
        return bs.morphology not in ["detached"]

    @staticmethod
    def kick_by_radius(bs):
        """
        Reference:

        https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/
        catalogue-of-stellar-parameters-from-the-detached-doublelined-eclipsing-binaries-in-the-milky-way/
        678AD993366CA4506DD671527906FCBD/core-reader
        """
        # 0.2 - 35 of Solar radii
        return not np.logical_and(np.less_equal(np.array([bs.primary.polar_radius, bs.secondary.polar_radius]) *
                                                (bs.semi_major_axis / SOLAR_RADIUS), 35),
                                  np.greater_equal(np.array([bs.primary.polar_radius, bs.secondary.polar_radius]) *
                                                   (bs.semi_major_axis / SOLAR_RADIUS), 0.2)).all()


class OverContactLCGenerator(LCGenerator):
    MORPHOLOGY = "over-contact"

    def __init__(self, phases, passband, threshold, **kwargs):
        if not np.all(np.array(kwargs["primary"]["surface_potential"]) ==
                      np.array(kwargs["secondary"]["surface_potential"])):
            raise ValueError("Primary and secondary potential boundaries have to be same")
        super().__init__(phases, passband, threshold, **kwargs)

    def eval_morphology_skip(self, bs):
        return bs.morphology not in ["over-contact"]

    @staticmethod
    def kick_by_potential(surface_potential, bs):
        return bs.primary.critical_surface_potential <= surface_potential
