### Observed lc table stored in

**url:** https://mega.nz/#!Gd1TlbwK!0Ow5qPOl5MPP0tsBaU3S0RSstv1i-9NVBj6FP-7kWHc


### Generator params
#### over-contact spot-free


#### detached spot-free

    PARMAS = {
        "system": {
            "inclination": (70.0, 95.0, 5.0),
            "period": (1.0, 40.0, 5)
        },
        "primary": {
            "mass": (0.2, 3.5, 0.5),
            "surface_potential": (2.0, 7.0, 1.0),
            "t_eff": (4000.0, 15000.0, 2000.0)
        },
        "secondary": {
            "mass": (0.2, 3.5, 0.5),
            "surface_potential": (2.0, 7.0, 1.0),
            "t_eff": (4000.0, 15000.0, 2000.0)
        }
    }
    
#### detached spotty

    PARMAS = {
        "system": {
            "inclination": (70.0, 95.0, 5.0),
            "period": (1.0, 40.0, 5) # higher density
        },
        "primary": {
            "mass": (0.2, 3.5, 0.5),
            "surface_potential": (2.0, 7.0, 1.0),
            "t_eff": (4000.0, 15000.0, 2000.0),
            "spots": [
                    {
                        "longitude": (0, 360),
                        "latitude": (0, 180),
                        "angular_radius": (10, 45),
                        "temperature_factor": (0.90, 1.10)
                    }
                ]
        },
        "secondary": {
            "mass": (0.2, 3.5, 0.5),
            "surface_potential": (2.0, 7.0, 1.0),
            "t_eff": (4000.0, 15000.0, 2000.0)
        }
    }

    
#### over-contact spot-free
    
    PARAMS =  {
        "system": {
            "inclination": (30.0, 95.0, 5.0),
            "period": (0.2, 0.9, 0.1)
        },
        "primary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (4000.0, 7000.0, 1000.0)
        },
        "secondary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (4000.0, 7000.0, 1000.0)
        }
    }

#### over-contact spotty

    PARMAS = {
        "system": {
            "inclination": (30.0, 95.0, 5.0),
            "period": (0.2, 0.9, 0.1)
        },
        "primary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (4000.0, 7000.0, 1000.0),
            "spots": [
                        {
                            "longitude": (0, 360),
                            "latitude": (0, 180),
                            "angular_radius": (10, 45),
                            "temperature_factor": (0.90, 1.10)
                        }
                    ]
        },
        "secondary": {
            "mass": (0.5, 3.25, 0.25),
            "surface_potential": (2.0, 4.0, 0.25),
            "t_eff": (4000.0, 7000.0, 1000.0)
        }
    }
