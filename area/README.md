Doing a quick check for Zeljko on the area of sky that is taken up by streaks

making a new conda env for everything

conda install numpy matplotlib jupyter

pip installing:
skyfield
ephem

pycraf failed on pip install, trying:
conda install pycraf -c conda-forge


----------------

ok, trying to create a pycraf env and work in there:

conda create -n pycraf-env python=3.6 pycraf

Looks like this will get me the openmp requirements, yay!

conda activate pycraf-env
pip install 
-----------------


VICTORY WITH:

ok, pip doesn't work with python 3.6, so trying things out with:

conda create -n pycraf-env python=3.7 pycraf
conda activate pycraf-env
pip install pyephem
pip install skyfield    
conda install jupyter

crap, I need some rubin sim stuff too
from rubin_sim dir:
conda install -c conda-forge --file=requirements.txt
pip install -e .

