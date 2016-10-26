# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import copy

# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)                       
# MEG only for now
def get_simu_data_ks(q,T, labels, outpath,
                  A, Q, Q0, Sigma_J_list, 
                  L_list_option = 0, L_list_param = None,
                  normalize_G_flag = False,
                  snr = 1.0):
    """
    Simulate stationary data with Kronecker cov, 
    using the mne sample subject, and freeserfer aparc 68 parcellation
    
    Lagged/non-stationary structure can be added later. 
    E.g. 2 time points. Use the common code structure
    
    Input:
        q, number of trials
        T, number of time points in each trial, 
            assume each time point = 10 ms
        labels, list of labels, length p
        outpath, full path of the output file name to write the simulated data,
                 no suffix of .mat or .fif
        A,[T,p,p], the time-varying autoregressive matrix
        Q, [p,p], PSD covariance of the ROI error
        Q0, [p,p], initial covariance of the ROI latent variables
        Sigma_J_list, [p+1]
        L_list_option: 0: all 1
                       1: sign flip +/-1 according to the eigen normal direction
                       2: svd weights
                       3: Gaussian, zero_mean
        L_list_param, scalar, used to control the prior covariance of L
        labels: list of label file paths, if None?
        normalize_G_flag, if True, normalize the G matrix, then scale J correspondingly
        snr = 1.0, sensor space SNR
        sensor_iir_flag, if True, use iir filter for sensor nosie, else, not
    Output: saved in a mat or npy file,
        M, [q, n, T]
        J, [q, m, T] true source data
        param, a dict of 
               Qu, Tcov, L_list, Sigma_J_list
        ROI_list,  list of ROI indices in the source space 
        fwd_path, 
        evoked_path, evoked template saved in a path?       
    """
    data_path = mne.datasets.sample.data_path()
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
    fwd = mne.pick_types_forward(fwd, meg=True, eeg=False) 
    G = fwd['sol']['data']
    m = G.shape[1]
    
    ind0 = fwd['src'][0]['inuse']
    ind1 = fwd['src'][1]['inuse']
    # positions of dipoles
    rr = np.vstack([fwd['src'][0]['rr'][ind0==1,:], 
                             fwd['src'][1]['rr'][ind1==1,:]])
    rr = rr/np.max(np.sum(rr**2, axis = 1))                       
    nn = np.vstack([fwd['src'][0]['nn'][ind0 == 1,:],
                    fwd['src'][1]['nn'][ind1 == 1,:]])
                    
    # normalize G
    source_weight = np.ones(m)
    if normalize_G_flag:
        source_weight =  np.sqrt(np.sum(G**2,axis = 0))
    
    G /= source_weight
    # J will be multipled by source_weight
    
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

    p = Q.shape[0]
    if p != n_ROI_valid or len(Sigma_J_list)-1!= p :
        raise ValueError("covariance size does not match ROI sets")
    
    # ==========generate L===============
    L_list = np.zeros(n_ROI_valid, dtype = np.object)
    if L_list_option == 0:
        # L_list fixed to 1
        for i in range(n_ROI_valid):
            tmp = np.ones( [len(ROI_list[i])])
            L_list[i] = tmp 
    elif L_list_option in [1,2]:
        for i in range(n_ROI_valid):
            tmp_nn = nn[ROI_list[i],:]
            tmpu, tmpd, tmpv = np.linalg.svd(tmp_nn)
            if L_list_option == 1:
                tmp= np.sign(np.dot(tmp_nn, tmpv[0]))
            else:
                tmp= tmpu[:,0]
            L_list[i] = tmp 
    elif L_list_option == 3:
        Q_L_list = np.zeros(n_ROI_valid, dtype = np.object)
        for i in range(n_ROI_valid):
            tmp_n = len(ROI_list[i])
            tmp_cov = np.zeros([tmp_n, tmp_n])
            for i0 in range(tmp_n):
                for i1 in range(tmp_n):
                    tmp_cov[i0,i1] = np.dot(nn[i0,:], nn[i1,:])* np.exp(-L_list_param * (np.sum((rr[i0,:]-rr[i1,:])**2)))
            tmp = np.random.multivariate_normal(np.zeros(tmp_n), tmp_cov)
            L_list[i] = tmp 
            Q_L_list[i] = tmp_cov
   
    # generate U
    u = np.zeros([q, p, T+1])
    for r in range(q):
        u[r,:,0] = np.random.multivariate_normal(np.zeros(p), Q0)
        for t in range(1,T+1):
            tmpA = A[t-1].copy() 
            u[r,:,t] = np.random.multivariate_normal(np.zeros(p), Q) + tmpA.dot(u[r,:,t-1])
           
            
    J = np.zeros([q,m,T+1])
    for i in range(n_ROI_valid):
        for j in range(len(ROI_list[i])):
            J[:,ROI_list[i][j],:] = L_list[i][j]*u[:,i,:] \
                  + np.random.randn(q, T+1)*np.sqrt(Sigma_J_list[i])
    if n_ROI_valid < n_ROI:
        i = -1
        J[:,ROI_list[i],:] = np.random.randn(q, len(ROI_list[i]), T+1)*np.sqrt(Sigma_J_list[i])

    J = (J.transpose([0,2,1])*source_weight).transpose([0,2,1])
    
    cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
    noise_cov = mne.read_cov(cov_fname)
    raw = mne.io.Raw(data_path + '/MEG/sample/sample_audvis_raw.fif')
    
    raw.info['bads'] = noise_cov['bads']
    picks = mne.pick_types(raw.info, meg=True)
    sel_cov = [l for l in range(len(noise_cov.ch_names)) \
          if l in picks \
          and noise_cov.ch_names[l] not in noise_cov['bads']]
    Sigma_E = noise_cov.data[:,sel_cov]
    Sigma_E = Sigma_E[sel_cov,:]
    # regularize the cov matrix
    Sigma_E *= np.eye(Sigma_E.shape[0])*0.01 + np.ones(Sigma_E.shape)
    Sigma_E_chol = np.linalg.cholesky(Sigma_E)
    G = fwd['sol']['data'][picks,:]
    n = G.shape[0]
    noise_M = np.dot(Sigma_E_chol, np.random.randn(q,n,(T+1)))
    M = (np.dot(G, J) + noise_M).transpose([1,0,2])
    

    # save the new regularized noise cov
    noise_cov_ch_names = [noise_cov.ch_names[l] for l in picks]
    noise_cov_new = mne.cov.Covariance(data = Sigma_E, names = noise_cov_ch_names,
                                       bads = noise_cov['bads'], projs = [], nfree = len(picks),
                                        eig = None, eigvec = None, method = None)
    # save the noise_cov
    cov_fname = "/home/ying/Dropbox/tmp/ROI_cov_simu/noise_cov-cov.fif"
    noise_cov_new.save(cov_fname)
    
    # save an evoked data
    tmin, tstep = 0, 0.01
    vertices = src[0]
    vertices = [ src[0]['vertno'], src[1]['vertno']]
    tmp_stc = mne.SourceEstimate(J[0], vertices = vertices, 
                                     tmin = tmin, tstep = tstep)
    evoked = mne.simulation.simulate_evoked(fwd, tmp_stc, raw.info, 
                                noise_cov, snr, iir_filter= None, verbose = False)
    evoked.info['bads'] = raw.info['bads'][0:1]
    evoked.save(outpath+"-ave.fif.gz")    

    mat_dict = dict(J = J, M = M, u = u, G = G,
                fwd_path = fwd_fname,
                ROI_list = ROI_list,
                A = A, Q = Q, Q0 = Q0,
                Sigma_J_list = Sigma_J_list,
                L_list = L_list,
                L_list_option = L_list_option,
                L_list_param = L_list_param if L_list_param is not None else 0.0,
                noise_cov_path = cov_fname, 
                Sigma_E = Sigma_E)
    scipy.io.savemat(outpath+".mat", mat_dict, oned_as = "row")

 
        
        
        

    

