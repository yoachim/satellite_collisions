import datetime
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import animation, rc

from astropy import time
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import EarthLocation
from pycraf import satellite
from lsst.sims.utils import Site
import skyfield.sgp4lib as sgp4lib
from astropy.time import Time


# adapting from:
# https://github.com/cbassa/satellite_analysis
# https://nbviewer.jupyter.org/github/yoachim/19_Scratch/blob/master/sat_collisions/bwinkel_constellation.ipynb


def satellite_mean_motion(altitude, mu=const.GM_earth, r_earth=const.R_earth):
    '''
    Compute mean motion of satellite at altitude in Earth's gravitational field.

    See https://en.wikipedia.org/wiki/Mean_motion#Formulae
    '''
    no = np.sqrt(4.0 * np.pi ** 2 * (altitude + r_earth) ** 3 / mu).to(u.day)
    return 1 / no


def tle_from_orbital_parameters(
        sat_name, sat_nr, epoch, inclination, raan, mean_anomaly, mean_motion):
    '''
    Generate TLE strings from orbital parameters.

    Note: epoch has a very strange format: first two digits are the year, next three
    digits are the day from beginning of year, then fraction of a day is given, e.g.
    20180.25 would be 2020, day 180, 6 hours (UT?)
    '''

    # Note: RAAN = right ascention (or longitude) of ascending node

    def checksum(line):
        s = 0
        for c in line[:-1]:
            if c.isdigit():
                s += int(c)
            if c == "-":
                s += 1
        return '{:s}{:1d}'.format(line[:-1], s % 10)

    tle0 = sat_name
    tle1 = checksum(
        '1 {:05d}U 20001A   {:14.8f}  .00000000  00000-0  50000-4 '
        '0    0X'.format(sat_nr, epoch))
    tle2 = checksum(
        '2 {:05d} {:8.4f} {:8.4f} 0001000   0.0000 {:8.4f} '
        '{:11.8f}    0X'.format(
            sat_nr, inclination.to_value(u.deg), raan.to_value(u.deg),
            mean_anomaly.to_value(u.deg), mean_motion.to_value(1 / u.day)
        ))

    return '\n'.join([tle0, tle1, tle2])


def create_constellation(altitudes, inclinations, nplanes, sats_per_plane):

    my_sat_tles = []
    sat_nr = 80000
    for alt, inc, n, s in zip(
            altitudes, inclinations, nplanes, sats_per_plane):

        if s == 1:
            # random placement for lower orbits
            mas = np.random.uniform(0, 360, n) * u.deg
            raans = np.random.uniform(0, 360, n) * u.deg
        else:
            mas = np.linspace(0.0, 360.0, s, endpoint=False) * u.deg
            mas += np.random.uniform(0, 360, 1) * u.deg
            raans = np.linspace(0.0, 360.0, n, endpoint=False) * u.deg
            mas, raans = np.meshgrid(mas, raans)
            mas, raans = mas.flatten(), raans.flatten()

        mm = satellite_mean_motion(alt)
        for ma, raan in zip(mas, raans):
            my_sat_tles.append(
                tle_from_orbital_parameters(
                    'TEST {:d}'.format(sat_nr), sat_nr, 19050.1,
                    inc, raan, ma, mm))
            sat_nr += 1

    return my_sat_tles


def starlink_constellation():
    altitudes = np.array([550, 1110, 1130, 1275, 1325, 345.6, 340.8, 335.9]) * u.km
    inclinations = np.array([53.0, 53.8, 74.0, 81.0, 70.0, 53.0, 48.0, 42.0]) * u.deg
    nplanes = np.array([72, 32, 8, 5, 6, 2547, 2478, 2493])
    sats_per_plane = np.array([22, 50, 50, 75, 75, 1, 1, 1])

    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane)
    my_sats = [satellite.get_sat(sat_tle) for sat_tle in my_sat_tles]

    return my_sats


time_J2000 = datetime.datetime(2000, 1, 1, 12, 0)


def _propagate(sat, dt):
    '''
    True equator mean equinox (TEME) position from `sgp4` at given time. Then converted to ITRS

    Parameters
    ----------
    sat : `sgp4.io.Satellite` instance
        Satellite object filled from TLE
    dt : `~datetime.datetime`
        Time
    Returns
    -------
    xs, ys, zs : float
        TEME (=True equator mean equinox) position of satellite [km]
    '''

    # pos [km], vel [km/s]
    position, velocity = sat.propagate(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

    if position is None:
        raise ValueError('Satellite propagation error')

    # I _think_ this is supposed to take time since J2000 in days?
    # looking at https://space.stackexchange.com/questions/25988/sgp4-teme-frame-to-j2000-conversion
    jd_ut1 = dt - time_J2000
    jd_ut1 = jd_ut1.days + jd_ut1.seconds/(3600.*24)
    new_position, new_velocity = sgp4lib.TEME_to_ITRF(jd_ut1, np.array(position), np.array(velocity)*86400)

    return tuple(new_position.tolist())


vec_propagate = np.vectorize(_propagate, excluded=['sat'], otypes=[np.float64] * 3)


def lsst_location():
    site = Site('LSST')
    obs_loc_lsst = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
    sat_obs_lsst = satellite.SatelliteObserver(obs_loc_lsst)
    return sat_obs_lsst

