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
import healpy as hp
import numbers
from scipy.spatial import cKDTree as kdTree

import os
import sqlite3


# adapting from:
# https://github.com/cbassa/satellite_analysis
# https://nbviewer.jupyter.org/github/yoachim/19_Scratch/blob/master/sat_collisions/bwinkel_constellation.ipynb

class FieldsDatabase(object):

    FIELDS_DB = "Fields.db"
    """Internal file containing the standard 3.5 degree FOV survey field
       information."""

    def __init__(self):
        """Initialize the class.
        """
        self.db_name = self.FIELDS_DB
        self.connect = sqlite3.connect(os.path.join(os.path.dirname(__file__),
                                       self.db_name))

    def __del__(self):
        """Delete the class.
        """
        self.connect.close()

    def get_field_set(self, query):
        """Get a set of Field instances.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        set
            The collection of Field instances.
        """
        field_set = set()
        rows = self.get_rows(query)
        for row in rows:
            field_set.add(tuple(row))

        return field_set

    def get_opsim3_userregions(self, query, precision=2):
        """Get a formatted string of OpSim3 user regions.

        This function gets a formatted string of OpSim3 user regions suitable
        for an OpSim3 configuration file. The format looks like
        (RA,Dec,Width):

        userRegion = XXX.XX,YYY.YY,0.03
        ...

        The last column is unused in OpSim3. The precision argument can be
        used to control the formatting, but OpSim3 configuration files use 2
        digits as standard.

        Parameters
        ----------
        query : str
            The query for field retrieval.
        precision : int, optional
            The precision used for the RA and Dec columns. Default is 2.

        Returns
        -------
        str
            The OpSim3 user regions formatted string.
        """
        format_str = "userRegion = "\
                     "{{:.{0}f}},{{:.{0}f}},0.03".format(precision)
        rows = self.get_rows(query)
        result = []
        for row in rows:
            result.append(format_str.format(row[2], row[3]))
        return str(os.linesep.join(result))

    def get_ra_dec_arrays(self, query):
        """Retrieve lists of RA and Dec.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        numpy.array, numpy.array
            The arrays of RA and Dec.
        """
        rows = self.get_rows(query)
        ra = []
        dec = []
        for row in rows:
            ra.append(row[2])
            dec.append(row[3])

        return np.array(ra), np.array(dec)

    def get_id_ra_dec_arrays(self, query):
        """Retrieve lists of fieldId, RA and Dec.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        numpy.array, numpy.array, numpy.array
            The arrays of fieldId, RA and Dec.
        """
        rows = self.get_rows(query)
        fieldId = []
        ra = []
        dec = []
        for row in rows:
            fieldId.append(int(row[0]))
            ra.append(row[2])
            dec.append(row[3])

        return np.array(fieldId, dtype=int), np.array(ra), np.array(dec)

    def get_rows(self, query):
        """Get the rows from a query.

        This function hands back all rows from a query. This allows one to
        perform other operations on the information than those provided by
        this class.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        list
            The set of field information queried.
        """
        cursor = self.connect.cursor()
        cursor.execute(query)
        return cursor.fetchall()



def read_fields():
    """
    Read in the Field coordinates
    Returns
    -------
    numpy.array
        With RA and dec in radians.
    """
    query = 'select fieldId, fieldRA, fieldDEC from Field;'
    fd = FieldsDatabase()
    fields = np.array(list(fd.get_field_set(query)))
    # order by field ID
    fields = fields[fields[:,0].argsort()]

    names = ['RA', 'dec']
    types = [float, float]
    result = np.zeros(np.size(fields[:, 1]), dtype=list(zip(names, types)))
    result['RA'] = np.radians(fields[:, 1])
    result['dec'] = np.radians(fields[:, 2])

    return result


def xyz_angular_radius(radius=1.75):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in degrees.

    Returns
    -------
    radius : float
    """
    return _xyz_angular_radius(np.radians(radius))


def _xyz_angular_radius(radius):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in radians.

    Returns
    -------
    radius : float
    """
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = _xyz_from_ra_dec(radius, 0)
    result = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    return result

def _xyz_from_ra_dec(ra, dec):
    """
    Utility to convert RA,dec positions in x,y,z space.

    Parameters
    ----------
    ra : float or array
        RA in radians
    dec : float or array
        Dec in radians

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    # It is ok to mix floats and numpy arrays.

    cosDec = np.cos(dec)
    return np.array([np.cos(ra) * cosDec, np.sin(ra) * cosDec, np.sin(dec)])



def _angularSeparation(long1, lat1, long2, lat2):
    """
    Angular separation between two points in radians

    Parameters
    ----------
    long1 is the first longitudinal coordinate in radians

    lat1 is the first latitudinal coordinate in radians

    long2 is the second longitudinal coordinate in radians

    lat2 is the second latitudinal coordinate in radians

    Returns
    -------
    The angular separation between the two points in radians

    Calculated based on the haversine formula
    From http://en.wikipedia.org/wiki/Haversine_formula
    """


    t1 = np.sin(lat2/2.0 - lat1/2.0)**2
    t2 = np.cos(lat1)*np.cos(lat2)*np.sin(long2/2.0 - long1/2.0)**2
    _sum = t1 + t2

    if isinstance(_sum, numbers.Number):
        if _sum<0.0:
            _sum = 0.0
    else:
        _sum = np.where(_sum<0.0, 0.0, _sum)

    return 2.0*np.arcsin(np.sqrt(_sum))


def _buildTree(ra, dec, leafsize=100, scale=None):
    """
    Build KD tree on simDataRA/Dec and set radius (via setRad) for matching.

    Parameters
    ----------
    ra, dec : float (or arrays)
        RA and Dec values (in radians).
    leafsize : int (100)
        The number of Ra/Dec pointings in each leaf node.
    scale : float (None)
        If set, the values are scaled up, rounded, and converted to integers. Useful for
        forcing a set precision and preventing machine precision differences
    """
    if np.any(np.abs(ra) > np.pi * 2.0) or np.any(np.abs(dec) > np.pi * 2.0):
        raise ValueError('Expecting RA and Dec values to be in radians.')
    x, y, z = _xyz_from_ra_dec(ra, dec)
    if scale is not None:
        x = np.round(x*scale).astype(int)
        y = np.round(y*scale).astype(int)
        z = np.round(z*scale).astype(int)
    data = list(zip(x, y, z))
    if np.size(data) > 0:
        try:
            tree = kdTree(data, leafsize=leafsize, balanced_tree=False, compact_nodes=False)
        except TypeError:
            tree = kdTree(data, leafsize=leafsize)
    else:
        raise ValueError('ra and dec should have length greater than 0.')

    return tree



def grow_hp(inmap, hpids, radius=1.75, replace_val=np.nan):
    """
    grow a healpix mask

    Parameters
    ----------
    inmap : np.array
        A HEALpix map
    hpids : array
        The healpixel values to grow around
    radius : float (1.75)
        The radius to grow around each point (degrees)
    replace_val : float (np.nan)
        The value to plug into the grown areas
    """
    nside = hp.npix2nside(np.size(inmap))
    theta, phi = hp.pix2ang(nside=nside, ipix=hpids)
    vec = hp.ang2vec(theta, phi)
    ipix_disc = [hp.query_disc(nside=nside, vec=vector, radius=np.radians(radius)) for vector in vec]
    ipix_disc = np.unique(np.concatenate(ipix_disc))
    outmap = inmap + 0
    outmap[ipix_disc] = replace_val
    return outmap


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
    sat_nr = 8000
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


def starlink_constellation(supersize=False, fivek=False):
    """
    Create a list of satellite TLE's
    """
    altitudes = np.array([550, 1110, 1130, 1275, 1325, 345.6, 340.8, 335.9])
    inclinations = np.array([53.0, 53.8, 74.0, 81.0, 70.0, 53.0, 48.0, 42.0])
    nplanes = np.array([72, 32, 8, 5, 6, 2547, 2478, 2493])
    sats_per_plane = np.array([22, 50, 50, 75, 75, 1, 1, 1])

    if supersize:
        # Let's make 4 more altitude and inclinations
        new_altitudes = []
        new_inclinations = []
        new_nplanes = []
        new_sat_pp = []
        for i in np.arange(0, 4):
            new_altitudes.append(altitudes+i*20)
            new_inclinations.append(inclinations+3*i)
            new_nplanes.append(nplanes)
            new_sat_pp.append(sats_per_plane)

        altitudes = np.concatenate(new_altitudes)
        inclinations = np.concatenate(new_inclinations)
        nplanes = np.concatenate(new_nplanes)
        sats_per_plane = np.concatenate(new_sat_pp)

    altitudes = altitudes * u.km
    inclinations = inclinations * u.deg
    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane, name='Starl')

    if fivek:
        stride = round(len(my_sat_tles)/5000)
        my_sat_tles = my_sat_tles[::stride]

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

        self.radius = xyz_angular_radius(fov)

    def _make_fields(self):
        """
        Make tesselation of the sky
        """
        # RA and dec in radians
        fields = read_fields()

        # crop off so we only worry about things that are up
        good = np.where(fields['dec'] > (self.alt_limit_rad - self.fov_rad))[0]
        self.fields = fields[good]

        self.fields_empty = np.zeros(self.fields.size)

        # we'll use a single tessellation of alt az
        leafsize = 100
        self.tree = _buildTree(self.fields['RA'], self.fields['dec'], leafsize, scale=None)

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
            try:
                sat.compute(self.observer)
            except ValueError:
                self.advance_epoch()
                sat.compute(self.observer)
            self.altitudes_rad.append(sat.alt)
            self.azimuth_rad.append(sat.az)
            self.eclip.append(sat.eclipsed)

        self.altitudes_rad = np.array(self.altitudes_rad)
        self.azimuth_rad = np.array(self.azimuth_rad)
        self.eclip = np.array(self.eclip)
        # Keep track of the ones that are up and illuminated
        self.above_alt_limit = np.where((self.altitudes_rad >= self.alt_limit_rad) & (self.eclip == False))[0]

    def fields_hit(self, mjd, fraction=False):
        """
        Return an array that lists the number of hits in each field pointing
        """
        mjds = mjd + self.tsteps
        result = self.fields_empty.copy()

        # convert the satellites above the limits to x,y,z and get the neighbors within the fov.
        for mjd in mjds:
            self.update_mjd(mjd)
            x, y, z = _xyz_from_ra_dec(self.azimuth_rad[self.above_alt_limit], self.altitudes_rad[self.above_alt_limit])
            if np.size(x) > 0:
                indices = self.tree.query_ball_point(np.array([x, y, z]).T, self.radius)
                final_indices = []
                for indx in indices:
                    final_indices.extend(indx)

                result[final_indices] += 1

        if fraction:
            n_hit = np.size(np.where(result > 0)[0])
            result = n_hit/self.fields_empty.size
        return result

    def check_pointing(self, pointing_alt, pointing_az, mjd):
        """
        See if a pointing has a satellite in it

        pointing_alt : float
           Altitude of pointing (degrees)
        pointing_az : float
           Azimuth of pointing (degrees)
        mjd : float
           Modified Julian Date at the start of the exposure

        Returns
        -------
        in_fov : float
            Returns the fraction of time there is a satellite in the field of view. Values >1 mean there were
            on average more than one satellite in the FoV. Zero means there was no satllite in the image the entire exposure.
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

    def look_ahead(self, pointing_alt, pointing_az, mjds):
        """
        Return 1 if satellite in FoV, 0 if clear
        """
        result = []
        for mjd in mjds:
            self.update_mjd(mjd)
            ang_distances = _angularSeparation(self.azimuth_rad[self.above_alt_limit], self.altitudes_rad[self.above_alt_limit],
                                               np.radians(pointing_az), np.radians(pointing_alt))
            if np.size(np.where(ang_distances <= self.fov_rad)[0]) > 0:
                result.append(1)
            else:
                result.append(0)
        return result

