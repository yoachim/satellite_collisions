import numpy as np
from lsst.sims.featureScheduler.utils import Constellation, starlink_constellation
import multiprocessing as mp



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

    constellations = [Constellation(chunk) for chunk in tle_chunks]

    pool = mp.Pool(n_cores)

    results = pool.starmap(p_check_pointing, [(constellation, 80., 0., 59853.8) for constellation in constellations])
    print(results)
    