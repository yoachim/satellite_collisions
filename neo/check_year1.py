import numpy as np
import pandas as pd
import sqlite3
from utils import Constellation, starlink_constellation
import sys


def check_pointings(night_max=366, dbfile='twilight_neo_mod2_v1.5_10yrs.db', outfile=None, supersize=False, fivek=False):
    """
    Check each pointing up to night
    """
    extra_fn = ''
    if supersize:
        extra_fn += 'supersize_'
    if fivek:
        extra_fn += 'fivek_'

    conn = sqlite3.connect(dbfile)
    df = pd.read_sql('select observationId, altitude, azimuth, observationStartMJD, night from summaryallprops where night <= %i order by observationId and note="twilight_neo"' % night_max, conn)
    conn.close()

    nobs = np.size(df['altitude'])

    names = ['observationId', 'hit']
    types = [int, float]
    hit = np.zeros(nobs, dtype=list(zip(names, types)))

    hit['observationId'] = df['observationStartMJD'].values

    sat_tles = starlink_constellation(supersize=supersize, fivek=fivek)
    constellation = Constellation(sat_tles, tstep=0.1, exptime=1.)

    for i, obs in df.iterrows():
        try:
            hit['hit'][i] = constellation.check_pointing(obs['altitude'], obs['azimuth'], obs['observationStartMJD'])
        except ValueError:
            constellation.advance_epoch()
            hit['hit'][i] = constellation.check_pointing(obs['altitude'], obs['azimuth'], obs['observationStartMJD'])
        progress = i/nobs*100
        text = "\rprogress = %.2f%%" % progress

        sys.stdout.write(text)
        sys.stdout.flush

    if outfile is None:
        outfile = 'hit_scale' + extra_fn + dbfile[:-3]
    np.save(outfile, hit)


if __name__ == '__main__':
    #check_pointings()
    check_pointings(supersize=True)
    #check_pointings(fivek=True)
