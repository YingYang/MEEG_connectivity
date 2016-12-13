# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import scipy.stats
import copy

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


outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
isMEG = True
MEGorEEG = ['EEG','MEG']


isMEG = True



pairs = [[0,0],[1,1],[0,1]]
#pairs = [[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]]
m1,m2 = 1,3
n_pairs = len(pairs)
    
mne_method = 'MNE'
ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g'] #, 'LO_c_g']
label_names = ['EVC','PPA']
p = len(ROI_bihemi_names)

old_new = ['old','new']
method_seq = ['mne','ks']

flag_cov_from_u = False
fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/roi_ks/Compare_Subj8/" 

fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
#fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"

subj = "Subj8"
result_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
# order, nips and new,  then dSPM and ks
#result_name =  [[result_dir + "nips2016_submission/Subj8_dspm_sol_bootstrp0.mat",
#                   result_dir + "/20161024_8000_source/Subj8_EVC_PPA_MEG_dspm_sol_bootstrp0.mat"],
#                 [result_dir + "nips2016_submission/Subj8_ks_sol_bootstrp0.mat",
#                   result_dir + "/20161024_8000_source/Subj8_EVC_PPA_MEG_ks_sol_bootstrp0.mat"]] 

result_name =  [[result_dir + "nips2016_submission/Subj8_dspm_sol_bootstrp0.mat",
                   result_dir + "/Subj8_EVC_PPA_ico41_%s_MEG_dspm_sol_bootstrp0.mat" %fname_suffix],
                 [result_dir + "nips2016_submission/Subj8_ks_sol_bootstrp0.mat",
                   result_dir + "/Subj8_EVC_PPA_ico41_%s_MEG_ks_sol_bootstrp0.mat" %fname_suffix]]                    
       

# debug

#=========== compare the boostraps 

# the two btstrp were different!
bstseq1 = scipy.io.loadmat(result_dir + "btstrp_seq.mat")
bstseq2 = scipy.io.loadmat(result_dir + "nips2016_submission/btstrp_seq.mat")



method = "dspm"
#result1 = scipy.io.loadmat(result_dir + "/20161025_local_test_aspect_and_wrong_noise_cov/Subj8_EVC_PPA_ico41_1_110Hz_notch_ica_ave_alpha15.0_no_aspect_MEG_%s_sol_bootstrp0.mat" %method )
#result2 = scipy.io.loadmat(result_dir + "/Subj8_EVC_PPA_ico41_1_110Hz_notch_ica_ave_alpha15.0_MEG_%s_sol_bootstrp0.mat" %method)
#result3 = scipy.io.loadmat(result_dir + "/20161024_8000_source/Subj8_EVC_PPA_MEG_%s_sol_bootstrp0.mat" %method)
#result0 = scipy.io.loadmat(result_dir + "nips2016_submission/Subj8_%s_sol_bootstrp0.mat" %method)






result2 = scipy.io.loadmat(result_dir + "/Subj8_EVC_PPA_ico41_1_110Hz_notch_ica_ave_alpha15.0_MEG_%s_sol_bootstrp1.mat" %method)
result0 = scipy.io.loadmat(result_dir + "nips2016_submission/Subj8_%s_sol_bootstrp1.mat" %method)
#result_list = [result1, result2, result3, result0]
#result_names = ['noaspect','withaspect','newoct6','nips']

result_list = [result2,  result0]
result_names = ['withaspect','nips']

for l in range(len(result_list)):
    result_list[l]['Q_hat']



i1,i2 = 1,1
plt.figure()
for l in range(len(result_list)):
    plt.plot(result_list[l]['A_hat'][:,i1,i2])
plt.legend(result_names)

trial_ind = 1  
ROI_ind = 1  
plt.figure()
for l in range(len(result_list)):
    plt.plot(result_list[l]['u_array_hat'][trial_ind, :,ROI_ind])
plt.legend(result_names)

for l in range(len(result_list)):
    plt.plot(result_list[l]['u_array_hat'][:, :,ROI_ind].std(axis = 0))
plt.legend(result_names)


# load the covariance 
cov0 = mne.read_cov(result_dir + "nips2016_submission/Subj8_STFT-R_MEGnoise_cov-cov.fif" ).data
cov1 = mne.read_cov(result_dir +  "Subj8_MEG_noise_cov-cov.fif").data
cov3 = mne.read_cov(result_dir +  "20161024_8000_source/Subj8_MEG_noise_cov-cov.fif").data

stft_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/"
cov00 = mne.read_cov(stft_dir +"Subj8_STFT-R_all_image_Layer_1_7_CCA_ncomp6_MEGnoise_cov-cov.fif").data

print "two noise cov cluster versus local"
np.linalg.norm(cov3-cov1)/np.linalg.norm(cov1)

print "two noise cov, stftr, local"
np.linalg.norm(cov0-cov1)/np.linalg.norm(cov1)

np.linalg.norm(cov00-cov1)/np.linalg.norm(cov1)

np.linalg.norm(cov00-cov0)/np.linalg.norm(cov0)

plt.plot(cov0.ravel(), cov1.ravel(), '.')
plt.plot(cov1.ravel(), cov3.ravel(), '.')

plt.plot(cov0.ravel(), cov00.ravel(), '.')


# what is wrong with the covariance term???
# why is the stft one different from the roi_cov one?
import mne
import scipy.io
import numpy as np
from copy import deepcopy

subj = "Subj8"
MEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
fname_suffix = MEG_fname_suffix 

"""
ave_mat_path =  MEG_data_dir + "epoch_raw_data/%s/%s_%s.mat" %(subj,  subj, fname_suffix)

datadir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"
datapath = datadir + "MEG_DATA/DATA/epoch_raw_data/%s/%s_%s.mat" \
                  %(subj, subj, fname_suffix)
epochs_pathA = (datadir + "MEG_DATA/DATA/epoch_raw_data/%s/%s_%s" \
   %(subj, subj, "run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"))

t_cov_baseline = -0.05
n_im = 362
# stft
mat_data = scipy.io.loadmat(datapath)
data = mat_data['ave_data'][:,:,:]
data -= np.mean(data,axis = 0)
    
n_times0 = data.shape[2]
print n_times0
epochsA = mne.read_epochs(epochs_pathA)
print epochsA.tmin
tmp_tmin = epochsA.tmin
tmp_tmax = 0.01*n_times0+tmp_tmin
# make sure the timing was correct!!
epochsA.crop(tmin = None, tmax = tmp_tmax)
n_timesA1 = len(epochsA.times)
epochsA1 = mne.epochs.concatenate_epochs([epochsA, deepcopy(epochsA)])
epochsA = epochsA1[0:n_im]
epochsA._data = data[:,:,0:n_timesA1].copy()
del(epochsA1)



epochs_pathB = MEG_data_dir + "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica-epo.fif.gz" %(subj,subj)

#mat_dict = scipy.io.loadmat(ave_mat_path)
#data = mat_dict['ave_data']
#picks_all = mat_dict['picks_all'][0]
#times = mat_dict['times'][0]
#data -= np.mean(data, axis = 0)
#del(mat_dict)

# make sure the epochs and the data have the same time length
n_times0 = data.shape[2]
print n_times0
         
# load data
epochs = mne.read_epochs(epochs_pathB)
print epochs.tmin
tmp_tmin = epochs.tmin
tmp_tmax = 0.01*n_times0+tmp_tmin
# make sure the timing was correct!!
epochs.crop(tmin = None, tmax = tmp_tmax)

n_times1 = len(epochs.times)

epochs1 = mne.epochs.concatenate_epochs([epochs, deepcopy(epochs)])
epochs = epochs1[0:n_im]
epochs._data = data[:,:,0:n_times1].copy()
del(epochs1)

np.linalg.norm(epochs._data - epochsA._data)/np.linalg.norm(epochs._data)   
    
# interplolate bad channels
epochs.interpolate_bads(reset_bads=True)



# a temporary comvariance matrix
# noise covariance was computed on the average across 3 to 6 repetitions, so no need to shrink it manually. 
noise_cov1 = mne.compute_covariance(epochsA, tmin=None, tmax=t_cov_baseline)
noise_cov2 = mne.compute_covariance(epochs, tmin=None, tmax=t_cov_baseline)
#2172 sample was used (6 time points x 362 trials), vs 18562 samples (51 time points used), not correct


epochsA.info['projs']
epochs.info['projs']
epochsA.info['projs'][0]['data']['data'] - epochs.info['projs'][0]['data']['data']

# epochs sampling rate was 1kHz, I did not downsample, so maybe that made things go wrong!

epochs.info['sfreq']
epochsA.info['sfreq']

"""
           

# try looking at Subj4 instead
"""
subj = "Subj4"
result_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
# order, nips and new,  then dSPM and ks
result_name =  [[result_dir + "Subj4_Jul22/Subj4_dspm_sol_bootstrp0.mat",
                   result_dir + "/Subj4_EVC_PPA_MEG_dspm_sol_bootstrp0.mat"],
                 [result_dir + "Subj4_Jul22/Subj4_ks_sol_bootstrp0.mat",
                   result_dir + "/Subj4_EVC_PPA_MEG_ks_sol_bootstrp0.mat"]]                    
"""                   


T0 = 79
A_est_all = np.zeros([2, 2, T0, p,p])
tildaA_all = np.zeros([2, 2, T0, T0])


pair_id = 2

n_bt = 1
T = 79
# i old versus new
# l method index
for i in range(2): 
    A_cell = np.zeros([2,n_bt,T,p,p])
    std_u = np.zeros([2, n_bt,T+1,p])
    tilda_A_entry = np.zeros([2,n_bt,n_pairs, T,T])
    i00 = 0
    for l in range(2):
        bootstrap_id = 0
        out_name = result_name[l][i]
        result = scipy.io.loadmat(out_name)
        A_cell[l,i00, :,:,:] = result['A_hat'][0:T]
        u_array_hat = result['u_array_hat'][:,0:T+1]
        std_u[l,i00] = np.std(u_array_hat, axis = 0)
        
        # compute u_i1 = \tilda A u_i2
        print "computing lagged A entries"
        tilde_A = np.zeros([T,T],dtype = np.object)  
        A = result['A_hat']
        for i0 in range(T):
            tilde_A[i0,i0] = A[i0].copy()
        for i0 in range(T):
            for j0 in range(i0+1,T):
                tmp = np.eye(p)
                for l0 in range(j0,i0-1,-1):
                    tmp = (np.dot(tmp, A[l0])).copy()
                tilde_A[i0,j0] = tmp
        for ii in range(n_pairs):
            for t1 in range(T):
                for t2 in range(T):
                    if t1<= t2:
                        tilda_A_entry[l,i00, ii,t1,t2] = tilde_A[t1,t2][pairs[ii][1], pairs[ii][0]]
                    else:
                        # region 2 -> region 1 lower diangonal feedback
                        tilda_A_entry[l,i00, ii,t1,t2] = tilde_A[t2,t1][pairs[ii][0], pairs[ii][1]]
        del(result)

    # plot the all parts in A, with bootstrap CI
    print "plotting A"
    A_est = A_cell[:,0]
    
    # save into the all matrix
    A_est_all[:,i,:,:,:] = A_est
    tildaA_all[:,i] = tilda_A_entry[:,0,pair_id,:,:] 
    
    var_u_est = std_u[:,0]**2

    alpha0 = scipy.stats.norm.ppf(1- 0.05/2)
    for l in range(2):
        plt.figure()
        count = 0
        ymin = None
        ymax = None
        for l1 in range(p):    
            # first dim is method
            _= plt.plot(var_u_est[l,:,l1])
            _= plt.xlabel('time (ms)')
        plt.title("%s %s" %(old_new[i],method_seq[l]))
            #_= plt.plot(times_in_ms[1::], np.zeros(T), 'k')
            #_= plt.ylim(ymin, ymax)
        #plt.legend(label_names)
        #fig_name = fig_outdir + "%s_%s_var_u.pdf" %(subj, method_string[l])
       
       
#
# i is old or new, l is method
plt.figure()
count = 1
for i in range(2):
    for l in range(2):
        plt.subplot(2,2,count);
        plt.plot(A_est_all[i,l,:,0,1]);
        plt.plot(A_est_all[i,l,:,1,0]);
        plt.plot(A_est_all[i,l,:,0,0]);
        plt.plot(A_est_all[i,l,:,1,1]);
        plt.title("%s %s" %(old_new[i],method_seq[l]))
        count += 1
        
        
    
   