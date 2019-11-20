import numpy as np
from lsst.sims.featureScheduler.utils import Constellation, starlink_constellation
import ipyparallel as ipp


# ipcluster start -n 4

def p_check_pointing(constellation, pointing_alt, pointing_az, mjd):
    """
    stupid function to help call things in parallel
    """
    result = constellation.check_pointing(pointing_alt, pointing_az, mjd)
    return result




if __name__ == "__main__":

    n_cores = 4

    tles = starlink_constellation()
    tle_chunks = np.array_split(tles, n_cores)
    tle_chunks = [chunk.tolist() for chunk in tle_chunks]

    # constellations = [Constellation(chunk) for chunk in tle_chunks]


    ## XXX--ok, this can't pickle things either. So I need to make the constellations 
    # on each worker, then fill them in with the tles

    rc = ipp.Client()

    
    # send a tle_chunk to each worker
    for view, chunk in zip(rc, tle_chunks):
        view['tle'] = chunk

    dview = rc[:]
    dview.execute("from lsst.sims.featureScheduler.utils import Constellation")
    dview.execute("constellation = Constellation(tle)")

    dview['alt'] = 80.
    dview['az'] = 0.
    dview['mjd'] = 59853.8 #22050.1

    dview.execute('temp = constellation.check_pointing(alt, az, mjd)')
    print(dview['temp'])
    
    dview.execute('temp = constellation.check_pointing(alt, az, mjd)')
    print(dview['temp'])


    dview['alt'] = 70.

    dview.execute('temp = constellation.check_pointing(alt, az, mjd)')
    print(dview['temp'])


    dview.execute('temp = constellation.check_pointing(alt, az, mjd)')
    print(dview['temp'])
    

    dview.execute('echeck = constellation.sat_list[0]._epoch')
    print(dview['echeck'])

    
    