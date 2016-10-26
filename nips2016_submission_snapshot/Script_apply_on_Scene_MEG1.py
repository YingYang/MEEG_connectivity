# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import copy

# this part to be optimized? chang it to a package?
if False:
    path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
    sys.path.insert(0, path)   
    import matplotlib
    matplotlib.use('Agg')                    
    import matplotlib.pyplot as plt
    #from get_simu_data_ks import get_simu_data_ks
    from get_estimate_baseline import get_estimate_baseline 
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
    labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/" 

# on cluster
path = "/home/yingyan1/source_roi_cov"
sys.path.insert(0, path)   
#from get_simu_data_ks import get_simu_data_ks
from get_estimate_baseline import get_estimate_baseline 
outdir = "/data2/tarrlab/MEG_NEIL/roi_ks/"
sys.path.insert(0, path)   
labeldir = "/data2/tarrlab/MEG_NEIL/roi_ks/ROI_labels/"


fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
fwd_path = "/data2/tarrlab/MEG_NEIL/MEG_preprocessed_data/fwd/"


subj_list = [1,2,3,4,5,6,7,8,9,10,12,13]
n_subj = len(subj_list)

B = 30
bootstrap_dict = scipy.io.loadmat(outdir + "btstrp_seq.mat")   
bootstrap_seq = bootstrap_dict['bootstrap_seq'].astype(np.int) 
bootstrap_seq = bootstrap_seq[0:B+1,:]

mne_method = 'MNE'
#for i0 in range(n_subj):
if True:
    # try subject 5
    i0 = 7
    subj = "Subj%d" %subj_list[i0]
    print subj
    #subj = "Subj7"

    ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g'] #, 'LO_c_g']
    nROI = len(ROI_bihemi_names)                    
    
    MEGorEEG = ['EEG','MEG']
    isMEG = True
 
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
    fwd_path = fwd_path + "%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
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
    if False:
        epochs_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"+ "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica-epo.fif.gz" %(subj,subj)
	noise_cov_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/" + "%s_STFT-R_MEGnoise_cov-cov.fif" %(subj)
	ave_mat_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"+ "%s/%s_%s" %(subj,  subj, "1_110Hz_notch_ica_ave_alpha15.0.mat")
	evoked_path = "/home/ying/dropbox_unsync/MEG_scene_neil/tmp/%s_tmp_evoked-ave.fif.gz" %subj
    # cluster version
    epochs_path = "/data2/tarrlab/MEG_NEIL/MEG_preprocessed_data/epoch_raw_data/" + "/%s/%s_run1_filter_1_110Hz_notch_ica-epo.fif.gz" %(subj,subj)
    noise_cov_path = outdir + "%s_STFT-R_MEGnoise_cov-cov.fif" %(subj)
    ave_mat_path = "/data2/tarrlab/MEG_NEIL/MEG_preprocessed_data/epoch_raw_data/"+ "%s/%s_%s" %(subj,  subj, "1_110Hz_notch_ica_ave_alpha15.0.mat")
    evoked_path = outdir + "%s_tmp_evoked-ave.fif.gz" %subj
                 
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
    
    epochs = mne.read_epochs(epochs_path)
    # interplolate bad channels
    epochs.interpolate_bads(reset_bads=True)
    evoked = epochs.average()
    evoked.save(evoked_path)
    
    prior_Q0, prior_Q, prior_sigma_J_list = None, None, None
    prior_A = dict(lambda0 = 0.0, lambda1 = 1.0) 
    depth = None
    force_fixed = True if depth is None else False
    MaxIter0, MaxIter = 100, 30
    tol0,tol = 1E-4,2E-2
    verbose0, verbose = False, False
    L_flag = False
    whiten_flag = True
    flag_A_time_vary = True  


    for bootstrp_id in range(B+1):
        print "bootstrap id = %d" %bootstrp_id
        #======== or use my hand-drawn labels, TBA
        out_name_ks = outdir + "%s_ks_sol_bootstrp%d.mat" % (subj, bootstrp_id)
        out_name_mne =  outdir + "%s_dspm_sol_bootstrp%d.mat" %(subj, bootstrp_id)  
        M0 = M[bootstrap_seq[bootstrp_id],:,:].copy()   
        get_estimate_baseline(M0, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_mne, 
                         method = mne_method, lambda2 = 1.0, prior_Q0 = prior_Q0, 
                         prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                         prior_A = prior_A, depth = depth, MaxIter0 = MaxIter0, 
                         MaxIter = MaxIter, tol0 = tol0, tol = tol, verbose0 = verbose, 
                         verbose = verbose, flag_A_time_vary = True,
                         flag_sign_flip = False, force_fixed=force_fixed) 
        #if bootstrp_id == 0:     
        if True:
            result_mne = scipy.io.loadmat(out_name_mne)
            p = len(labels)
            T = len(times_in_ms)-1
            result_mne_gamma0 = np.linalg.cholesky(result_mne['Q0_hat'])
            result_mne_gamma = np.linalg.cholesky(result_mne['Q_hat'])
            result_mne_A = result_mne['A_hat']
            result_mne_sigma_J_list = np.sqrt(result_mne['Sigma_J_list_hat'][0])
 	    del(result_mne)
            ini_Gamma0_list = [result_mne_gamma0]
            ini_A_list = [result_mne_A] 
            ini_Gamma_list = [result_mne_gamma]
            ini_sigma_J_list = [result_mne_sigma_J_list]
            flag_inst_ini = False
        if False:
            # if bootstrapped, start with the initial one, in case we find a different local min
            result_ks0 =  scipy.io.loadmat(outdir + "%s_ks_sol_bootstrp%d.mat" %(subj, 0))
            ini_Gamma0_list = [np.linalg.cholesky(result_ks0['Q0_hat'])]
            ini_A_list = [result_ks0['A_hat']] 
            ini_Gamma_list = [np.linalg.cholesky(result_ks0['Q_hat']) ]
            ini_sigma_J_list = [ np.sqrt(result_ks0['Sigma_J_list_hat'][0])]
            flag_inst_ini = False
        #====================================================================== 
        if True:
            out_name_ks = out_name_ks  
            from get_estimate_ks import get_estimate_ks   
            print "bootstrap_id %d" % bootstrp_id
            get_estimate_ks(M0, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_ks, 
                             prior_Q0 = prior_Q0, prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                             prior_A = prior_A,
                             depth = depth, MaxIter0 = MaxIter0, MaxIter = MaxIter,
                             tol0 = tol0, tol = tol,
                             verbose0 = verbose0, verbose = verbose, verbose_coarse =False,
                             L_flag = L_flag, whiten_flag = whiten_flag, n_ini = -1, 
                             flag_A_time_vary = flag_A_time_vary, use_pool = False, 
                             MaxIter_coarse = 3, ini_Gamma0_list = ini_Gamma0_list,
                             ini_A_list = ini_A_list, ini_Gamma_list = ini_Gamma_list,
                             ini_sigma_J_list = ini_sigma_J_list, force_fixed=force_fixed, 
                             flag_inst_ini = flag_inst_ini, a_ini = 0.1)

