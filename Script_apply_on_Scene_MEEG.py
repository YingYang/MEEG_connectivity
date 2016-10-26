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

# other rois?
ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g']
                 #, 'RSC_c_g', 'TOS_c_g','LOC_c_g']
mne_method = 'MNE'


if Flag_on_cluster:
    MEG_data_dir = "/data2/tarrlab/MEG_NEIL/MEG_preprocessed_data/"
    EEG_data_dir = "/data2/tarrlab/MEG_NEIL/EEG_preprocessed_data/"
else:
    MEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
    EEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"


fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
flag_ico_4 = True
outname_suffix = "EVC_PPA_ico4%d_%s" % (flag_ico_4, fname_suffix)

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
B = 15
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
time_ind = np.all(np.vstack([times >= -0.05, times <= 0.8 ]), axis = 0)
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

for bootstrp_id in range(B):
    print "bootstrap id = %d" %bootstrp_id
    #======== or use my hand-drawn labels, TBA
    out_name_ks = outdir + "%s_%s_%s_ks_sol_bootstrp%d.mat" \
                 % (subj, outname_suffix, MEGorEEG[isMEG], bootstrp_id)
    out_name_mne =  outdir + "%s_%s_%s_dspm_sol_bootstrp%d.mat" \
                 %(subj, outname_suffix, MEGorEEG[isMEG], bootstrp_id)  
    M0 = M[bootstrap_seq[bootstrp_id],:,:].copy()   
    # initial value set by MNE
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
        T = M.shape[-1]-1
        result_mne_gamma0 = np.linalg.cholesky(result_mne['Q0_hat'])
        result_mne_gamma = np.linalg.cholesky(result_mne['Q_hat'])
        result_mne_A = result_mne['A_hat']
        result_mne_sigma_J_list = np.sqrt(result_mne['Sigma_J_list_hat'][0])
        A_0_identity = np.zeros([T,p,p])
        for t in range(T):
            A_0_identity[t] = np.eye(p)
        del(result_mne)
 
        ini_Gamma0_list = [result_mne_gamma0]
        ini_A_list = [result_mne_A] 
        ini_Gamma_list = [result_mne_gamma]
        ini_sigma_J_list = [result_mne_sigma_J_list]
        flag_inst_ini = False
    """
    if False:
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
                     MaxIter_coarse = 3, ini_Gamma0_list = ini_Gamma0_list,
                     ini_A_list = ini_A_list, ini_Gamma_list = ini_Gamma_list,
                     ini_sigma_J_list = ini_sigma_J_list, force_fixed=force_fixed, 
                     flag_inst_ini = flag_inst_ini, a_ini = 0.1)



""" 
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/roi_ks/"                 
                
                 
#for i0 in range(n_subj):
    subj = "Subj%d" %subj_list[i0]   
    out_name_ks = outdir + "%s_ks_sol_bootstrp%d.mat" % (subj,0)
    out_name_mne =  outdir + "%s_dspm_sol_bootstrp%d.mat" % (subj,0)
           
    result_ks = scipy.io.loadmat(out_name_ks)                 
    result_mne = scipy.io.loadmat(out_name_mne)
    # visualize A_hat
    '''
    vmin, vmax = 0,1.0
    #vmin, vmax = None, None
    m1,m2 = 6,6
    tmpT = len(times_in_ms)-1
    #A_list = [result_ks['A_hat'], result_mne['A_hat']]
    for l in range(len(A_list)):
        plt.figure()
        count = 0
        for t in range(0,tmpT,3):
            _ = plt.subplot(m1,m2,count+1); 
            _ = plt.imshow( np.abs(A_list[l][t]), vmin = vmin, vmax = vmax, interpolation = "none", aspect = "auto")
            count += 1
            _ = plt.colorbar()
            _ = plt.title("%1.2f s" % times[t])
        plt.tight_layout()
   
    pair = [0,1]
    for l in range(len(A_list)):
        # EVC-PPA vs PPA-EVC
        plt.figure()
        normalizer = np.sqrt(np.abs(A_list[l][:,pair[0], pair[0]]*A_list[l][:,pair[1], pair[1]]))
        #normalizer = np.ones(T)
        tmp0 = A_list[l][:,pair[0], pair[1]]/normalizer
        tmp1 = A_list[l][:,pair[1], pair[0]]/normalizer
        plt.plot(times_in_ms[1::], tmp0);
        plt.plot(times_in_ms[1::], tmp1);
        plt.legend([label_names[pair[1]]+ "->"+label_names[pair[0]],
                    label_names[pair[0]]+ "->"+label_names[pair[1]]])
    '''    
    
    method_string = ['ks','mne']
    result_list = [result_ks, result_mne]
    for l in range(2):
        result = result_list[l]
        plt.figure()
        count = 0
        for l1 in range(p):
            for l2 in range(p):
                _= plt.subplot(p,p, count+1);
                # first dim is method
                _= plt.errorbar(times_in_ms[1::], result['A_hat'][:,l1,l2])
                print l1, l2
                _= plt.xlabel('time (ms)')
                _= plt.title('A[:,%d,%d]'% (l1,l2))
                count += 1
        fig_name = fig_outdir + "%s_%s_A.pdf" %(subj, method_string[l])
        plt.savefig(fig_name)
        plt.close('all')
        
        
    # consider every pair of two regions, all time lags
    flag_cov_from_u = False
    #result_names = ['ks','mne']
    #result_list = [result_ks, result_mne]
    #result_list = [result_mne]
    result_names = method_string
    pairs = [[0,0],[1,1],[0,1]]
    #pairs = [[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]]
    n_pairs = len(pairs)
    
    for j in range(len(result_list)):
        result = result_list[j]
        from ROI_Kalman_smoothing import get_cov_u
        T = len(times_in_ms)-1
        u_array_hat = result['u_array_hat'] # q,T+1, p
        plt.figure(); plt.plot(times_in_ms, np.std(u_array_hat, axis = 0)); 
        plt.legend(label_names);
        plt.xlabel('time (ms)'); plt.ylabel("std of ROI means")
        fig_name = fig_outdir + "%s_%s_std_u.pdf" %(subj, result_names[j])
        plt.savefig(fig_name)
        if flag_cov_from_u:       
            cov_u = np.cov(u_array_hat.reshape([u_array_hat.shape[0], -1]).T)    
        else:
            cov_u = get_cov_u(result['Q0_hat'], result['A_hat'], result['Q_hat'], T, flag_A_time_vary = True) # pT x pT, p first 
        paired_lags = np.zeros([n_pairs, (T+1), (T+1)])
        p = len(labels)
        for i in range(n_pairs):
            for t1 in range(T+1):
                for t2 in range(T+1):
                    paired_lags[i,t1,t2] = cov_u[t1*p+pairs[i][0], t2*p+pairs[i][1]]\
                    /np.sqrt(cov_u[t1*p+pairs[i][0], t1*p+pairs[i][0]])/ np.sqrt(cov_u[t2*p+pairs[i][1], t2*p+pairs[i][1]]) 
        
        
        
        inv_cov_u = np.linalg.inv(cov_u)
        paired_lags_inv = np.zeros([n_pairs, (T+1), (T+1)])
        p = len(labels)
        for i in range(n_pairs):
            for t1 in range(T+1):
                for t2 in range(T+1):
                    paired_lags_inv[i,t1,t2] = inv_cov_u[t1*p+pairs[i][0], t2*p+pairs[i][1]]
        
        
        
        data_names = ["margcorr"]#,"invcov"] 
        data_list =  [paired_lags]#, paired_lags_inv] 
        for j1 in range(len(data_list)):
            data = data_list[j1]
            plt.figure(figsize =(14,8))
            #tmp_vmax = np.max(data)
            tmp_vmax = None
            for i in range(n_pairs):
                _= plt.subplot(2, n_pairs//2+1,i+1)
                _=plt.imshow(np.abs(data[i]), interpolation = "none", aspect = "auto", 
                           extent = [times_in_ms[0], times_in_ms[-1], 
                        times_in_ms[0], times_in_ms[-1]],origin= "lower", vmin = 0, vmax = tmp_vmax)
                _=plt.xlabel(label_names[pairs[i][0]]); plt.ylabel(label_names[pairs[i][1]])
                _=plt.colorbar()
                
            plt.tight_layout()
            fig_name = fig_outdir + "%s_%s_%s_sample_cov%d.pdf" %(subj, result_names[j], data_names[j1], flag_cov_from_u)
            plt.savefig(fig_name)
            plt.close()
"""                
