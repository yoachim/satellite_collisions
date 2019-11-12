import numpy as np
import pandas as pd
import sqlite3
from utils import Constellation, starlink_constellation
import sys



def check_pointings(night_max=366, dbfile='baseline_v1.3_10yrs.db', outfile=None):
    """
    Check each pointing up to night
    """

    conn = sqlite3.connect(dbfile)
    df = pd.read_sql('select observationId, altitude, azimuth, observationStartMJD, night from summaryallprops where night <= %i order by observationId' % night_max, conn)
    conn.close()

    nobs = np.size(df['altitude'])

    names = ['observationId', 'hit']
    types = [int, float]
    hit = np.zeros(nobs, dtype=list(zip(names, types)))

    hit['observationId'] = df['observationStartMJD'].values

    sat_tles = starlink_constellation()
    constellation = Constellation(sat_tles)

    for i, obs in df.iterrows():
        try:
            hit['hit'][i] = constellation.check_pointing(obs['altitude'], obs['azimuth'], obs['observationStartMJD'])
        except:
            constellation.advance_epoch()
            hit['hit'][i] = constellation.check_pointing(obs['altitude'], obs['azimuth'], obs['observationStartMJD'])
        progress = i/nobs*100
        text = "\rprogress = %.1f%%" % progress

        sys.stdout.write(text)
        sys.stdout.flush

    if outfile is None:
        outfile = 'hit_' + dbfile[:-3]
    np.save(outfile, hit)


if __name__ == '__main__':
    check_pointings(night_max=2)
