# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import copy

# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)                       
import matplotlib.pyplot as plt
#from get_simu_data_ks import get_simu_data_ks
from get_simu_data_ks_ico_4 import get_simu_data_ks_ico_4


from get_estimate_baseline import get_estimate_baseline 
from get_estimate_ks import get_estimate_ks 



                 
