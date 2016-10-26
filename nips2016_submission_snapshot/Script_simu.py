# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import copy

# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)
from ROI_cov_Kronecker import sample_kron_cov                        
import matplotlib.pyplot as plt
from get_simu_data import get_simu_data
from get_estimate import get_estimate

anat_ROI_list_fname = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/aparc_ROI_names.npy"
#========
#data_path = mne.datasets.sample.data_path()
#subjects_dir = data_path + '/subjects'
#labels = mne.read_labels_from_annot('sample', parc='aparc',
#                                    subjects_dir=subjects_dir)
#ROI_names = list()
#for label in labels:
#    ROI_names.append(label.name)
#np.save(ana_ROI_list_fname, ROI_names)
#=================================
#anat_ROI_names = np.load(anat_ROI_list_fname)
#p = 10
#anat_ROI_names = anat_ROI_names[np.random.choice(range(len(anat_ROI_names)), p, replace = False)]
p = 6
anat_ROI_names= ['lateralorbitofrontal-lh','lateralorbitofrontal-rh',
                 'parahippocampal-lh', 'parahippocampal-rh',
                 'lateraloccipital-lh', 'lateraloccipital-rh']

#========== QUcov=======
alpha = 5.0
p = len(anat_ROI_names)
tmp = np.random.randn(p,p)
r = np.random.gamma(shape=0.5, scale=1.0, size=p)
QUcov = np.dot(tmp*r, (tmp*r).T)
QUcov += np.eye(p)
diag = np.sqrt(np.diag(QUcov))
denom = np.outer(diag, diag)
QUcov = QUcov/denom* alpha

scale_factor = 1E-9
QUcov = QUcov*scale_factor**2

#=========Tcov=======
T = 5
a0,b0 = 1.0, 1E-1 # a exp (-b ||x-y||^2)
# Gaussian process kernel for temporal smoothness
Tcov = np.zeros([T,T])
for i in range(T):
    for j in range(T):
        Tcov[i,j] = a0 * np.exp(-b0 * (i-j)**2)
Tcov += 0.01*np.eye(T)
print np.linalg.cond(Tcov)

#========Sigma_J_list ===
Sigma_J_list = np.random.gamma(shape=2, scale=1.0, size= p+1)
Sigma_J_list = Sigma_J_list*scale_factor**2
#========
# generate the simulation data
outpath = "/home/ying/Dropbox/tmp/ROI_cov_simu/%d_ROIs_simu0.mat" %(p)
q = 320
get_simu_data(q,T, anat_ROI_names, outpath,
                            QUcov, Tcov, Sigma_J_list, 
                            L_list_option = 0,
                            L_list_param = None,
                            normalize_G_flag = False,
                            snr = 1.0, sensor_iir_flag = False)
                            
#=======solution                           
             
if False:               
    filepath = outpath
    method = "ROIcov"
    outname = "/home/ying/Dropbox/tmp/ROI_cov_simu/%d_ROIs_simu0_%s.mat" %(p, method)
    get_estimate(filepath, outname, method = "ROIcov", 
                     loose = None, depth = 0.8,
                     verbose = True)    
                     
                     
    plt.figure()
    QU_list = [Qu_hat, QUcov]
    for l in range(len(QU_list)):
        plt.subplot(1,2,l+1)
        plt.imshow(QU_list[l], interpolation = "none")
        plt.colorbar()
    
    diag0 = np.sqrt(np.diag(QUcov))
    denom = np.outer(diag0, diag0)
    corr_true = np.abs(QUcov/denom)        
    Corr_list = [corr_hat, corr_true]
    for l in range(len(Corr_list)):
        plt.subplot(1,2,l+1)
        plt.imshow(Corr_list[l], interpolation = "none")

