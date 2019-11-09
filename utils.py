import datetime
import numpy as np


from astropy import time
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import EarthLocation
from pycraf import satellite
from lsst.sims.utils import Site
import skyfield.sgp4lib as sgp4lib
from astropy.time import Time
import ephem
from lsst.sims.utils import _angularSeparation


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


def tle_from_orbital_parameters(sat_name, sat_nr, epoch, inclination, raan,
                                mean_anomaly, mean_motion):
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


def create_constellation(altitudes, inclinations, nplanes, sats_per_plane, epoch=22050.1, name='Test'):

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
                    name+' {:d}'.format(sat_nr), sat_nr, epoch,
                    inc, raan, ma, mm))
            sat_nr += 1

    return my_sat_tles


def starlink_constellation():
    """
    Create a list of satellite TLE's
    """
    altitudes = np.array([550, 1110, 1130, 1275, 1325, 345.6, 340.8, 335.9]) * u.km
    inclinations = np.array([53.0, 53.8, 74.0, 81.0, 70.0, 53.0, 48.0, 42.0]) * u.deg
    nplanes = np.array([72, 32, 8, 5, 6, 2547, 2478, 2493])
    sats_per_plane = np.array([22, 50, 50, 75, 75, 1, 1, 1])

    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane, name='Starlink')

    return my_sat_tles


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


class Constellation(object):
    """
    Have a class to hold ephem satellite objects

    Parameters
    ----------
    sat_tle_list : list of str
        A list of satellite TLEs to be used
    tstep : float (5)
        The time step to use when computing satellite positions in an exposure
    """

    def __init__(self, sat_tle_list, alt_limit=30., fov=3.5, tstep=5., exptime=30.):
        self.sat_list = [ephem.readtle(tle.split('\n')[0], tle.split('\n')[1], tle.split('\n')[2]) for tle in sat_tle_list]
        self.alt_limit_rad = np.radians(alt_limit)
        self.fov_rad = np.radians(fov)
        self._make_observer()
        self._make_fields()
        self.tsteps = np.arange(0, exptime+tstep, tstep)/3600./24.  # to days

    def _make_fields(self):
        """
        Make tesselation of the sky
        """

        # Read in the fields, build a kd-tree
        pass

    def _make_observer(self):
        telescope = Site(name='LSST')

        self.observer = ephem.Observer()
        self.observer.lat = telescope.latitude_rad
        self.observer.lon = telescope.longitude_rad
        self.observer.elevation = telescope.height

    def advance_epoch(self, advance=100):
        """
        Advance the epoch of all the satellites
        """

        # Because someone went and put a valueError where there should have been a warning
        # I prodly present the hackiest kludge of all time
        for sat in self.sat_list:
            sat._epoch += advance

    def update_mjd(self, mjd):
        """
        observer : ephem.Observer object
        """

        self.observer.date = ephem.date(time.Time(mjd, format='mjd').datetime)

        self.altitudes_rad = []
        self.azimuth_rad = []
        self.eclip = []
        for sat in self.sat_list:
            sat.compute(self.observer)
            self.altitudes_rad.append(sat.alt)
            self.azimuth_rad.append(sat.az)
            self.eclip.append(sat.eclipsed)

        self.altitudes_rad = np.array(self.altitudes_rad)
        self.azimuth_rad = np.array(self.azimuth_rad)
        self.eclip = np.array(self.eclip)
        # Keep track of the ones that are up and illuminated
        self.above_alt_limit = np.where((self.altitudes_rad >= self.alt_limit_rad) & (self.eclip == False))[0]

    def fraction_covered(self, mjd):
        """
        Return the fraction of fields that would include a satellite at this time
        """
        mjds = mjd + self.tsteps

        # convert the satellites above the limits to x,y,z and get the neighbors within the fov.
        pass

    def check_pointing(self, pointing_alt, pointing_az, mjd):
        """
        See if a pointing has a satellite in it

        pointing_alt : float
           Altitude of pointing (degrees)
        pointing_az : float
           Azimuth of pointing (degrees)
        mjd : float
           Modified Julian Date at the start of the exposure
        """

        mjds = mjd + self.tsteps
        in_fov = 0

        for mjd in mjds:
            self.update_mjd(mjd)
            ang_distances = _angularSeparation(self.azimuth_rad[self.above_alt_limit], self.altitudes_rad[self.above_alt_limit],
                                               np.radians(pointing_az), np.radians(pointing_alt))
            in_fov += np.size(np.where(ang_distances <= self.fov_rad)[0])
        in_fov = in_fov/mjds.size
        return in_fov
