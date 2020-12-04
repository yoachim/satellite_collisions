import numpy as np
from lsst.sims.almanac import Almanac
import matplotlib.pylab as plt
import pandas as pd
import sqlite3
from utils import Constellation, starlink_constellation
import sys


def night_fractions(length=366, supersize=False, fourk=False):
    """
    Find the fraction of fields that have a satellite in them
    """

    mjd_start = 59853.8

    mjd_end = mjd_start + length

    mjd_check = np.arange(mjd_start, mjd_end+0.75, 0.75)

    alm = Almanac(mjd_start=mjd_start)

    sat_tles = starlink_constellation(supersize=supersize, fourk=fourk)
    constellation = Constellation(sat_tles)

    night_report = alm.get_sunset_info(mjd_start)
    current_night = night_report['night']

    result_fractions = []
    result_mjds = []
    for mjd in mjd_check:
        night_report = alm.get_sunset_info(mjd)
        if night_report['night'] != current_night:
            middle = (night_report['sun_n18_setting'] + night_report['sun_n18_rising'])/2.
            mjds = [night_report['sun_n12_setting'], (night_report['sun_n12_setting']+night_report['sun_n18_setting'])/2.,
                    night_report['sun_n18_setting'], middle-1./24, middle,
                    middle+1./24, night_report['sun_n18_rising'],
                    (night_report['sun_n12_rising']+night_report['sun_n18_rising'])/2.,
                    night_report['sun_n12_rising']]
            result = [constellation.fields_hit(mjd_in_night, fraction=True) for mjd_in_night in mjds]
            result.append(current_night)
            result_fractions.append(result)
            result_mjds.append(mjds)
            text = "\rnight = %i" % current_night
            sys.stdout.write(text)
            sys.stdout.flush

        current_night = night_report['night']
    return result_fractions, result_mjds


if __name__ == '__main__':
    #supersize = True
    supersize = False
    fourk = True
    result_fractions, result_mjds = night_fractions(supersize=supersize, fourk=fourk)
    extra = ''
    if supersize:
        extra = 'super'
    if fourk:
        extra='4k'
    np.savez('contam_fractions_scale'+extra+'.npz', result_fractions=result_fractions, result_mjds=np.array(result_mjds))
