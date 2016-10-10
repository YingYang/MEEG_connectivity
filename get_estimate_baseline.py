# -*- coding: utf-8 -*-
import numpy as np
import sys
import scipy.io

import mne
import copy

# no depth weithing is reapplied to the source solution

# for portability, I need to make write it as a package
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)
from ROI_Kalman_smoothing import (get_param_given_u)

# use standard MNE/LCMV to obtain the A matrix
#==============================================================================
def get_estimate_baseline(M, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name, 
                 method = "dSPM", lambda2 = 1.0,                 
                 prior_Q0 = None, prior_Q = None, prior_sigma_J_list = None, 
                 prior_A = None,
                 depth = 0.8,
                 MaxIter0 = 100, MaxIter = 50,
                 tol0 = 1E-4, tol = 1E-2,
                 verbose0 = True, verbose = False, flag_A_time_vary = False,
                 flag_sign_flip = False, force_fixed=True):
    """
    Inputs: 
        M, [q, n_channels, n_times] sensor data
        ROI_list, ROI indices list
        fwd_path, full path of the forward solution
        evoked_path, full path of the evoked template
        noise_cov_path, full path of the noise covariance
        out_name, full path of the mat name to save
        
        priors:
        prior_Q0, prior_Q, prior_sigma_J_list, not implemented, may be inverse gamma or gamma
        prior_A, dict(lambda0 = 0.0, lambda1 = 1.0)
        
        depth: forward weighting parameter
        verbose:
        whiten_flag: if True, whiten the data, so that sensor error is identity  
        n_ini, number of random initializations
        
        TBA
    """
    
    q,_,T0 = M.shape
    T = T0-1
    # this function returns a list, take the first element
    evoked = mne.read_evokeds(evoked_path)[0]
    # depth weighting, TO BE MODIFIED
    print force_fixed 
    fwd0 = mne.read_forward_solution(fwd_path, force_fixed=force_fixed, surf_ori = True)
    fwd= copy.deepcopy(fwd0)
    noise_cov = mne.read_cov(noise_cov_path)
    
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
                    
    ch_names = evoked.info['ch_names']
    # create the epochs first?
    M_all = np.zeros([q, len(ch_names), T0])
    valid_channel_ind = [i for i in range(len(ch_names)) if ch_names[i] not in evoked.info['bads'] ]
    M_all[:,valid_channel_ind, :] = M.copy()    
    events = np.ones([M.shape[0],3], dtype = np.int)                
    epochs = mne.EpochsArray(data = M_all, info = evoked.info, events = events,
                             tmin =  evoked.times[0], event_id = None, reject = None)

    # if method is MNE
    if method in ['MNE','dSPM','sLORETA']:
        # create inverse solution
        # get the ROI_flipped data
        inv_op = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov,
                   loose = 0.0, depth = depth,fixed = True)
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2 = lambda2,
                     method = method)
        
        #roi_stc = mne.extract_label_time_course(stcs, labels, fwd['src'], mode = "mean_flip")
    elif method in ['LCMV']:
        # why this time window?
        data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                  method='shrunk')
        stcs = list()
        for r in range(q):
            tmp_evoked = epochs[r].average()                          
            stcs.append(mne.beamformer.lcmv(tmp_evoked, fwd, noise_cov, data_cov, reg=0.01,
               pick_ori= 'normal'))                       

    p = n_ROI_valid                                                
    ROI_U = np.zeros([q,p,T0])
    Sigma_J_list_hat = np.zeros(len(ROI_list))
    for i in range(p):
        tmp_ind = ROI_list[i]
        # svd of nn
        tmp_nn = nn[tmp_ind,:]
        tmpu,tmpd, tmpv = np.linalg.svd(tmp_nn)
        signs = np.sign(np.dot(tmp_nn, tmpv[0,:])) if flag_sign_flip else np.ones(len(tmp_ind))
        for r in range(q):
            ROI_U[r,i,:] = np.sum(stcs[r].data[tmp_ind].T*signs, axis = 1)
            Sigma_J_list_hat[i] += np.var( (stcs[r].data[tmp_ind].T*signs).T - ROI_U[r,i,:])
        Sigma_J_list_hat[i] /= np.float(q)
    
    for r in range(q):
        tmp = stcs[r].data[ROI_list[-1]]
        Sigma_J_list_hat[-1] += np.var(tmp)
    Sigma_J_list_hat[-1] /= np.float(q)
    
    # maybe also compare with the mne solution
    #roi_stc = mne.extract_label_time_course(stcs, labels[i], fwd['src'], mode = "mean_flip")
    
    u_array = ROI_U.transpose([0,2,1])
    Gamma0_0 = np.eye(p)
    A_0 = np.zeros([T,p,p])
    Gamma_0 = np.eye(p)
    # first run the non_prior version to get a global solution
    Gamma0_1, A_1, Gamma_1 = get_param_given_u(u_array, Gamma0_0, A_0, Gamma_0, 
       flag_A_time_vary = flag_A_time_vary,
       prior_Q0 = None, prior_A = None, prior_Q = None,
       MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0,
       MaxIter = MaxIter, tol = tol, verbose = verbose)
       
    # compute the Q0, A, Q, with priors
    Gamma0_hat, A_hat, Gamma_hat = get_param_given_u(u_array, Gamma0_1, A_1, Gamma_1, 
       flag_A_time_vary = flag_A_time_vary,
       prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
       MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0,
       MaxIter = MaxIter, tol = tol, verbose = verbose)
     
    Q0_hat = Gamma0_hat.dot(Gamma0_hat.T)
    Q_hat = Gamma_hat.dot(Gamma_hat.T)
    result = dict(Q0_hat = Q0_hat, A_hat = A_hat, Q_hat = Q_hat,
                  method = method, lambda2 = lambda2, u_array_hat = u_array,
                  Sigma_J_list_hat = Sigma_J_list_hat)
    scipy.io.savemat(out_name, result)

    
            
            