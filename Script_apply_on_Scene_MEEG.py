# usage: python Script_apply_on_Scene_MEEG Subj1 MEG
import numpy as np
import mne
import sys
import scipy.io
import copy
import getpass
import os

import matplotlib
matplotlib.use('Agg')                    
import matplotlib.pyplot as plt

username = getpass.getuser()
Flag_on_cluster = True if username == "yingyan1" else False
if Flag_on_cluster:
    paths = ["/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"]
else:
    paths = ["/home/yingyan1/source_roi_cov/"]
for l in range(len(paths)):
    sys.path.insert(0,paths[l])
 

from get_estimate_ks import get_estimate_ks   
from get_estimate_baseline import get_estimate_baseline 


#bootstrap_id_seq = [0]
bootstrap_id_seq = range(9, 10)
MEGorEEG = ["EEG","MEG"]

MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect"

# debug:
#MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
#EEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"

 
subj = str(sys.argv[1])
modality = str(sys.argv[2])

isMEG = True if modality == "MEG" else False
print "subj = %s, isMEG %d" %(subj, isMEG)
# usage python Script_use_stftr.py Subj1

if Flag_on_cluster:
    outdir = "/data2/tarrlab/MEG_NEIL/roi_ks/"
    labeldir = "/data2/tarrlab/MEG_NEIL/ROI_labels/" 
else:
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
    labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/" 



ROI_bihemi_names = ['pericalcarine', 'PPA_c_g', 'TOS_c_g', 'RSC_c_g', 'LOC_c_g',
                            'medialorbitofrontal'] 
ROI_names = ['EVC','PPA','TOS','RSC','LOC','mOFC'] 
mne_method = 'dSPM'


if Flag_on_cluster:
    MEG_data_dir = "/data2/tarrlab/MEG_NEIL/MEG_preprocessed_data/"
    EEG_data_dir = "/data2/tarrlab/MEG_NEIL/EEG_preprocessed_data/"
else:
    MEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
    EEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"


fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
flag_ico_4 = False
outname_suffix = "%dROI_ico4%d_%s" % (len(ROI_names), flag_ico_4, fname_suffix)

if isMEG:
    if not flag_ico_4:
        fwd_path = MEG_data_dir + "fwd/%s/%s_ave-fwd.fif" %(subj, subj)
    else:
        fwd_path = MEG_data_dir + "fwd/%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
        
    # must use the smoothed here, otherwise the sampling rate is wrong 1000 vs 100
    epochs_path = MEG_data_dir + "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz" %(subj,subj)
    ave_mat_path =  MEG_data_dir + "epoch_raw_data/%s/%s_%s.mat" %(subj,  subj, fname_suffix)
else:
    if not flag_ico_4:
        fwd_path = EEG_data_dir + "%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
    else:
        fwd_path = EEG_data_dir + "%s_EEG/%s_EEG_ico-4-fwd.fif" %(subj, subj)
    
    # must use the smootehd here, other wise the sampling rate is wrong    
    epochs_path = EEG_data_dir + "epoch_raw_data/%s_EEG/%s_EEG_filter_1_110Hz_notch_PPO10POO10_swapped_ica_reref_smoothed-epo.fif.gz" %(subj,subj)
    ave_mat_path =  EEG_data_dir + "epoch_raw_data/%s_EEG/%s_EEG_%s.mat" %(subj,  subj, fname_suffix)          
                                   

# temporary files for quicker computation
# evoked template and noise covariance
evoked_path = outdir+ "%s_%s_tmp_evoked-ave.fif.gz" %(subj, MEGorEEG[isMEG])  
noise_cov_path =  outdir+"%s_%s_noise_cov-cov.fif" %(subj, MEGorEEG[isMEG])   


# if there is a bootstrap sequence, keep using it
# predefiend sequence
btstrap_path = outdir + "btstrp_seq.mat"
if os.path.isfile(btstrap_path):
    print "bootstrap seq exits!"
    pass
else:
    n_im = 362
    if not os.path.isfile(btstrap_path):
        B = 50
        original_order = (np.arange(0,n_im)).astype(np.int)
        bootstrap_seq = np.zeros([B+1, n_im], dtype = np.int)
        bootstrap_seq[0] = original_order
        for l in range(B):
            bootstrap_seq[l+1] = (np.random.choice(original_order, n_im)).astype(np.int)
        scipy.io.savemat(btstrap_path, dict(bootstrap_seq = bootstrap_seq))


#=================== load the btstrap sequence ================================
# all subjects shared the same one?
B = 30
bootstrap_dict = scipy.io.loadmat(btstrap_path)   
bootstrap_seq = bootstrap_dict['bootstrap_seq'].astype(np.int) 
bootstrap_seq = bootstrap_seq[0:B+1,:]



#================== define the ROIs=============================                   
# read ROI labels
labeldir1 = labeldir + "%s/labels/" % subj
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
# keep fixed orientation                 
fwd = mne.read_forward_solution(fwd_path, force_fixed=True, surf_ori=True)
m = fwd['sol']['ncol']

src = fwd['src']
ROI_list0 = list() 
for i in range(len(labels)):
    _, sel = mne.source_space.label_src_vertno_sel(labels[i],src)
    ROI_list0.append(sel)

ROI_list = copy.deepcopy(ROI_list0)
# check if the ROIs has overlap, if so, move the overlap to non-ROI 
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        tmp_intersect = np.intersect1d(ROI_list0[i], ROI_list0[j])
        if len(tmp_intersect) > 0:
            print "intersection between %d and %d found, removing it" %(i,j)
            print tmp_intersect
            ROI_list[i] = np.setdiff1d(ROI_list0[i], tmp_intersect)
            ROI_list[j] = np.setdiff1d(ROI_list0[j], tmp_intersect)
    
ROI0_ind = np.arange(0, m, dtype = np.int)    
for i in range(len(labels)):
    ROI0_ind = np.setdiff1d(ROI0_ind, sel)
ROI_list.append(ROI0_ind)
n_ROI = len(ROI_list)
n_ROI_valid = n_ROI-1

#  ============prep operations to for the data=============================
n_im = 362           
mat_dict = scipy.io.loadmat(ave_mat_path)
data = mat_dict['ave_data']
picks_all = mat_dict['picks_all'][0]
times = mat_dict['times'][0]
data -= np.mean(data, axis = 0)
del(mat_dict)

# make sure the epochs and the data have the same time length
n_times0 = data.shape[2]
print n_times0
          
       
if not ( os.path.isfile(epochs_path) and os.path.isfile(noise_cov_path)):
    # load data
    epochs = mne.read_epochs(epochs_path)
    print epochs.tmin
    tmp_tmin = epochs.tmin
    tmp_tmax = 0.01*n_times0+tmp_tmin
    # make sure the timing was correct!!
    epochs.crop(tmin = None, tmax = tmp_tmax)
    
    n_times1 = len(epochs.times)
    
    if isMEG:
        epochs1 = mne.epochs.concatenate_epochs([epochs, copy.deepcopy(epochs)])
        epochs = epochs1[0:n_im]
        epochs._data = data[:,:,0:n_times1].copy()
        del(epochs1)
    else:
        epochs = epochs[0:n_im]
        epochs._data = data[:,:,0:n_times1].copy()
            
    # interplolate bad channels
    epochs.interpolate_bads(reset_bads=True)
    evoked = epochs.average()
    evoked.save(evoked_path)
    t_cov_baseline = -0.05
    noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=t_cov_baseline)
    noise_cov.save(noise_cov_path)


#offset = 0.04 if isMEG else 0.00
time_ind = np.all(np.vstack([times >= -0.05, times <= 0.9 ]), axis = 0)
M = data[:,:, time_ind]
#times_in_ms = (times[time_ind]-offset)*1000.0
#print len(times_in_ms)   


#========= actual computation including bootstrap===============================
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



for bootstrp_id in bootstrap_id_seq:
#if True:
    print "bootstrap id = %d" %bootstrp_id
    #======== or use my hand-drawn labels, TBA
    out_name_ks = outdir + "%s_%s_%s_ks_sol_bootstrp%d.mat" \
                 % (subj, outname_suffix, MEGorEEG[isMEG], bootstrp_id)
    out_name_mne =  outdir + "%s_%s_%s_%s_sol_bootstrp%d.mat" \
                 %(subj, outname_suffix, MEGorEEG[isMEG], mne_method, bootstrp_id)  
    M0 = M[bootstrap_seq[bootstrp_id],:,:].copy()   
    # initial value set by MNE
    if False:
        get_estimate_baseline(M0, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_mne, 
                     method = mne_method, lambda2 = 1.0, prior_Q0 = prior_Q0, 
                     prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                     prior_A = prior_A, depth = depth, MaxIter0 = MaxIter0, 
                     MaxIter = MaxIter, tol0 = tol0, tol = tol, verbose0 = verbose, 
                     verbose = verbose, flag_A_time_vary = True,
                     flag_sign_flip = False, force_fixed=force_fixed) 
    #if bootstrp_id == 0:     
    if True:
        print "ks"
        result_mne = scipy.io.loadmat(out_name_mne)
        p = len(labels)
        T = M.shape[-1]-1
        scale = 1E-9
        result_baseline_A = result_mne['A_hat']
        result_baseline_gamma0 = np.linalg.cholesky(result_mne['Q0_hat'])*scale
        result_baseline_gamma = np.linalg.cholesky(result_mne['Q_hat'])*scale
        
        result_baseline_sigma_J_list = np.sqrt(result_mne['Sigma_J_list_hat'][0])*scale
        A_0_identity = np.zeros([T,p,p])
        for t in range(T):
            A_0_identity[t] = np.eye(p)
        del(result_mne)
 
        ini_Gamma0_list = [result_baseline_gamma0, np.eye(p)*scale]
        ini_A_list = [result_baseline_A, result_baseline_A] 
        ini_Gamma_list = [result_baseline_gamma, np.eye(p)*scale]
        ini_sigma_J_list = [result_baseline_sigma_J_list, np.ones(p)*scale]
        flag_inst_ini = False
        """
        # if bootstrapped, start with the initial one, in case we find a different local min
        result_ks0 =  scipy.io.loadmat(outdir + "%s_ks_sol_bootstrp%d.mat" %(subj, 0))
        ini_Gamma0_list = [np.linalg.cholesky(result_ks0['Q0_hat'])]
        ini_A_list = [result_ks0['A_hat']] 
        ini_Gamma_list = [np.linalg.cholesky(result_ks0['Q_hat']) ]
        ini_sigma_J_list = [ np.sqrt(result_ks0['Sigma_J_list_hat'][0])]
        flag_inst_ini = False
        """
    #====================================================================== 
        print "bootstrap_id %d" % bootstrp_id
        get_estimate_ks(M0, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_ks, 
                         prior_Q0 = prior_Q0, prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                         prior_A = prior_A,
                         depth = depth, MaxIter0 = MaxIter0, MaxIter = MaxIter,
                         tol0 = tol0, tol = tol,
                         verbose0 = verbose0, verbose = verbose, verbose_coarse =False,
                         L_flag = L_flag, whiten_flag = whiten_flag, n_ini = -1, 
                         flag_A_time_vary = flag_A_time_vary, use_pool = False, 
                         MaxIter_coarse = 1, ini_Gamma0_list = ini_Gamma0_list,
                         ini_A_list = ini_A_list, ini_Gamma_list = ini_Gamma_list,
                         ini_sigma_J_list = ini_sigma_J_list, force_fixed=force_fixed, 
                         flag_inst_ini = flag_inst_ini, a_ini = 0.1)


"""


"""
