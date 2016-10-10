# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import copy
import os.path
# this part to be optimized? chang it to a package?
#path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
#sys.path.insert(0, path)                       
# MEG only for now
from shutil import copyfile

# note this funciton can only be run on tarrlabb434 due to data path 
#$anat_ROI_names= ['pericalcarine-lh', 'pericalcarine-rh',
                 #'lateraloccipital-lh', 'lateraloccipital-rh',
#                'parahippocampal-lh', 'parahippocampal-rh']
#                 'medialorbitofrontal-lh','medialorbitofrontal-rh']

subj = "Subj12"
# use the Scene_MEG_EEG subject as example source space
fwd1_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/fwd/" \
                + "%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
                
                
ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g']
nROI = len(ROI_bihemi_names)                    
labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/" 
MEGorEEG = ['EEG','MEG']
isMEG = True
# For now for MEG only
stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
            + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"

# read ROI labels
labeldir1 = labeldir + "%s/" % subj
# load and merge the labels
labels_bihemi = list()
lr_label_list = list()
lr_label_names = list()
for j in range(len(ROI_bihemi_names)):
    for hemi in ['lh','rh']:
        print subj, j, hemi
        tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj, ROI_bihemi_names[j],hemi)
        tmp_label = mne.read_label(tmp_label_path)
        lr_label_list.append(tmp_label)
        lr_label_names.append(ROI_bihemi_names[j]+"_" + hemi)
for j in range(len(ROI_bihemi_names)):
    labels_bihemi.append(lr_label_list[2*j]+lr_label_list[2*j+1]) 


labels = labels_bihemi                                
fwd = mne.read_forward_solution(fwd1_fname, force_fixed=True, surf_ori=True )
print "difference of G"
print np.max(np.abs(fwd['sol']['data']))/np.min(np.abs(fwd['sol']['data']))
 
m = fwd['sol']['ncol']    
ind0, ind1 = fwd['src'][0]['inuse'], fwd['src'][1]['inuse']
# positions of dipoles
rr = np.vstack([fwd['src'][0]['rr'][ind0==1,:], 
               fwd['src'][1]['rr'][ind1==1,:]])
rr = rr/np.max(np.sum(rr**2, axis = 1))                       
nn = np.vstack([fwd['src'][0]['nn'][ind0 == 1,:],
                fwd['src'][1]['nn'][ind1 == 1,:]])

#========label index =============
src = fwd['src']
ROI_list = list() 
ROI0_ind = np.arange(0, m, dtype = np.int)
for i in range(len(labels)):
    _, sel = mne.source_space.label_src_vertno_sel(labels[i],src)
    ROI_list.append(sel)
    ROI0_ind = np.setdiff1d(ROI0_ind, sel)
ROI_list.append(ROI0_ind)
n_ROI = len(ROI_list)
n_ROI_valid = n_ROI-1
     
outpath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/ROI_map/"+ "%s_EVC_PPA_" %subj
ROI_stc = np.ones([m,1])
for i in range(n_ROI_valid):
    ROI_stc[ROI_list[i]] = i+2.0
print np.max(ROI_stc)
vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
stc = mne.SourceEstimate(data = ROI_stc,vertices = vertices, 
                             tmin =0.0, tstep = 1.0 )
stc.save(outpath+"ROI-stc")  
print "ROI stc saved"
    
   