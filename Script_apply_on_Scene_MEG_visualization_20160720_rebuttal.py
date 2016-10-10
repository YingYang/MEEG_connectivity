# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import scipy.stats
import copy

from mne.forward import  is_fixed_orient, _to_fixed_ori
from mne.inverse_sparse.mxne_inverse import _prepare_gain

# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)   
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#matplotlib.use('Agg')                    
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'plasma'
#from get_simu_data_ks import get_simu_data_ks
from ROI_Kalman_smoothing import get_cov_u

subj_list = [1,2,3,4,5,6,7,8,9,10,12,13]
n_subj = len(subj_list)
outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
pairs = [[0,0],[1,1],[0,1]]
#pairs = [[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]]
m1,m2 = 1,3
n_pairs = len(pairs)
    

mne_method = 'MNE'
ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g'] #, 'LO_c_g']
#for i0 in range(n_subj):

p = len(ROI_bihemi_names)

method_string = ['ks','dspm']
flag_cov_from_u = False
fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/roi_ks/"  

label_names = ['EVC','PPA']
p = len(label_names)
if True: 
    # ==================variance explained====================================
    
    # load the G and L
    i0 = 3
    subj = "Subj%d" %subj_list[i0]
    print subj
    #subj = "Subj7"

    ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g'] #, 'LO_c_g']
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
    
    flag_merge_bilateral = True
    labels = labels_bihemi if flag_merge_bilateral else lr_label_list
    label_names = ROI_bihemi_names if flag_merge_bilateral else lr_label_names
    
    # define ROI list
    fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/fwd/" \
                        + "%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
    # keep fixed orientation                 
    fwd = mne.read_forward_solution(fwd_path, force_fixed=True, surf_ori=True)
    m = fwd['sol']['ncol']
    
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

    #  inpute parameters
    epochs_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"+\
                    "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica-epo.fif.gz" %(subj,subj)
    noise_cov_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/" \
                        + "%s_STFT-R_MEGnoise_cov-cov.fif" %(subj)
    ave_mat_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"\
                     + "%s/%s_%s" %(subj,  subj, "1_110Hz_notch_ica_ave_alpha15.0.mat")
    evoked_path = "/home/ying/dropbox_unsync/MEG_scene_neil/tmp/%s_tmp_evoked-ave.fif.gz" %subj
                 
    mat_dict = scipy.io.loadmat(ave_mat_path)
    ave_data = mat_dict['ave_data']
    picks_all = mat_dict['picks_all'][0]
    times = mat_dict['times'][0]
    ave_data -= np.mean(ave_data, axis = 0)
    del(mat_dict)
    
    offset = 0.04
    time_ind = np.all(np.vstack([times >= -0.06, times <= 0.74 ]), axis = 0)
    M = ave_data[:,:, time_ind]
    times_in_ms = (times[time_ind]-offset)*1000.0
    print len(times_in_ms)

    
    whiten_flag = True
    depth = 0.0
    q,_,T0 = M.shape
    T = T0-1
    # this function returns a list, take the first element
    evoked = mne.read_evokeds(evoked_path)[0]
    # depth weighting, TO BE MODIFIED
    force_fixed=True
    print force_fixed
    fwd0 = mne.read_forward_solution(fwd_path, force_fixed= force_fixed, surf_ori = True)
    fwd= copy.deepcopy(fwd0)
    noise_cov = mne.read_cov(noise_cov_path)
    Sigma_E = noise_cov.data
    
    # orientation of dipoles
    ind0 = fwd['src'][0]['inuse']
    ind1 = fwd['src'][1]['inuse']
    # positions of dipoles
    rr = np.vstack([fwd['src'][0]['rr'][ind0==1,:], 
                             fwd['src'][1]['rr'][ind1==1,:]])
    rr = rr/np.max(np.sum(rr**2, axis = 1))                       
    nn = np.vstack([fwd['src'][0]['nn'][ind0 == 1,:],
                    fwd['src'][1]['nn'][ind1 == 1,:]])
    # number of dipoles                
    #m = rr.shape[0]
    all_ch_names = evoked.ch_names
    sel = [l for l in range(len(all_ch_names)) if all_ch_names[l] not in evoked.info['bads']]
    print "fixed orient"
    print is_fixed_orient(fwd)
    if not is_fixed_orient(fwd):
        _to_fixed_ori(fwd)
    print "difference of G"
    print np.max(np.abs(fwd['sol']['data']))/np.min(np.abs(fwd['sol']['data']))
    
    if whiten_flag:
        pca = True
        G, G_info, whitener, source_weighting, mask = _prepare_gain(fwd, evoked.info,
                                                                    noise_cov, pca =pca,
                                                                    depth = depth, loose = None,
                                                                    weights = None, weights_min = None)
        #Sigma_E_chol = np.linalg.cholesky(Sigma_E)
        #Sigma_E_chol_inv = np.linalg.inv(Sigma_E_chol)
        #G = np.dot(Sigma_E_chol_inv, G)
        # after whitening, the noise cov is assumed to identity
        #Sigma_E = (np.dot(Sigma_E_chol_inv, Sigma_E)).dot(Sigma_E_chol_inv.T)
        Sigma_E = np.eye(G.shape[0])
        M = (np.dot(whitener, M)).transpose([1,0,2])
    else:
        G = fwd['sol']['data'][sel,:]
        G_column_weighting = (np.sum(G**2, axis = 0))**(depth/2)
        G = G/G_column_weighting
    
    
    y_array = M.transpose([0,2,1]) # q,T,n    
    scale_factor = 1E-9
    p = n_ROI_valid
    
    L_list_0 = list()
    for i in range(n_ROI_valid):
        L_list_0.append(np.ones(ROI_list[i].size))
        
    m = G.shape[1]
    L = np.zeros([m, n_ROI_valid])
    for i in range(n_ROI_valid):
        L[ROI_list[i], i] = L_list_0[i]
    GL = G.dot(L)        
    
    for l in range(2):
        bootstrap_id = 0
        out_name = outdir + "%s_%s_sol_bootstrp%d.mat" % (subj, method_string[l],bootstrap_id)
        result = scipy.io.loadmat(out_name)
        u_array_hat = result['u_array_hat']
        
        y_hat = np.zeros(M.shape)
        for r in range(q):
            y_hat[r] = GL.dot(u_array_hat[r].T)
        
        var_prop = 1.0-np.mean((M-y_hat)**2) \
                      /np.mean((M-M.mean(axis = 0))**2)
        print "%1.1f%%" %(var_prop*100.0)
        print method_string[l]
               
            
            
            
           
   