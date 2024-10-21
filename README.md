# SPCAndie

Here is my code to filter out the telluric in W1, the first component wPCA of the differential line width (dLW) data from lbl.

This notebook book is analysing a toy model of this contamination on a simulated EV Lac data. 
The analysis on true data will be published in an in comming paper (Charpentier et al. in prep.).

Feel free to use this code on your own research (but citing me will be nice of you :) ).

![SPCAndie logo](https://github.com/Paul-Charpentier/SPCAndie/blob/main/SPCAndie_logo.png)

# Getting Started 

Here is a tutorial to run the code in one single line of command (like as a black-box). I suggest you to check the `Tutorial.ipynb` for a step by step tutorial on how does it works. 

Download the `SPCAndie.py` file and then place it in your current-work-directory. 
This directory should be organized as follow : 
```
. 
├── lblrv/
│   └── TARGET_TEMPLATE/ 
│       ├── 1234567o_pp_e2dsff_tcorr_AB_TARGET_TEMPLATE_lbl.fits  #all your individual lblrv files
│       ├── ...
│       └── ...
|   └── ..._.../
│       ├── ...
│       └── ...
├── out/                                                          # the output directory
└── wapiti/                                                       # your wapiti installation 
```

Then you will probably have to install all the dependencies : 

```
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from wpca import PCA, WPCA, EMPCA
from astropy.timeseries import LombScargle
import random
import pacmap
import matplotlib
from scipy import stats
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import sys
from wapiti import wapiti_tools, wapiti
from astropy.table import Table
from collections import Counter
from tqdm import tqdm
import gc
```


Then finally you can edit the `## Set-Up` block (this is the only part you are suppose to edit unless you know what you are doing).

In that block you will have first to set the list of your targets and their rotation period accordong to the litterature : 
```
Target = ['EV_LAC', 'GL410']  # The list of your targets
Prot = [4.3615, 13.87]        # The list of their rotation period in days
Template = Target             # The list of the lbl templates for each traget (usually its the same star as the target, thus Template = Target )
```

Then you have to specify wether or not you want the code to tell you what is it doing rn or if you want it to draw the plots and save it in your output folder. 

```
chatty = True
plotty = True
```

It can happend that you have outliers in your data, we remove it as in Ould-Elhkim et al. (2023), the following lines are there to configurate the outliers rejection (this default configuration should be fine for most of the stars): 
```
initial_mad_rejection = 10  # MAD rejection threshold for the outlier removal
n_component_rejection = 10  # Number of component to check for the outlier removal
looplier = 10               # Number of outlier rejection loops
```

Then you define the period grid on which the code will compute the periodogram of each lines, we chose the log uniform distribution to have a fair distribution between short an long periodicities: 
```
yr = 365.25
period_grid = np.logspace(np.log(1.25), 3, 1000)
frequency = 1/period_grid # periodogram frequency grid
```

Finally, indicates the path of the current-work-directory as defined earlier : 

```
cwd = "/your/current/work/directory"
```

Save it and run it using `python3 SPCAndie.py` in your console command
