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
def get_simu_data_ks_ico_4(q,T, labels, outpath,
                  A, Q, Q0, Sigma_J_list, 
                  L_list_option = 0, L_list_param = None,
                  flag_time_smooth_source_noise = False, 
                  flag_space_smooth_source_noise = False,
                  flag_nn_dot = False,
                  subj = "Subj2"):
    """
    Following the model assumption, with no vialations
    # weird, Subj1 has really crazy G, one column is larger than any other 
    # because I saved and loaded free-orientation fwd?
    """
    # use the Scene_MEG_EEG subject as example source space
    fwd1_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/fwd/" \
                    + "%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
                   
    # even with file copy , still problematic really weird               
    #if not os.path.isfile(outpath+"-fwd.fif"):
        #  this one is free orientation, after saving and reloading it, G would change
        
        # copy the forward file to the outpath-fwd.fif
        # I can not save and load it. saving it result in extremely weird G matrix
        # No idea why it is the case. Maybe I should ask in mne-python mailing list
    #    copyfile(fwd1_fname, outpath+"-fwd.fif")

    #fwd = mne.read_forward_solution(outpath+"-fwd.fif", force_fixed=True, surf_ori=True )'
    fwd = mne.read_forward_solution(fwd1_fname, force_fixed=True, surf_ori=True )
    print "difference of G"
    print np.max(np.abs(fwd['sol']['data']))/np.min(np.abs(fwd['sol']['data']))
    
    
    # this noise cov might be singular, create a regularized one from Raw
    #cov_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/" \
    #                + "%s_STFT-R_MEGnoise_cov-cov.fif" %(subj)
    cov_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/" \
                    + "%s_STFT-R_all_image_Layer_1_7_CCA_ncomp6_MEGnoise_cov-cov.fif" %(subj)
    noise_cov = mne.read_cov(cov_fname) 

    # obtain evoked info
    evoked_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"+\
                "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz" %(subj,subj)
    evoked = mne.read_epochs(evoked_path)
    info = evoked.info
    del(evoked)
    
    noise_cov_reg = mne.cov.regularize(noise_cov, info, proj = False)
    del(noise_cov)
    # load empty room raw
    #empty_room_raw_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"+\
    #            "filtered_raw_data/%s/%s_emptyroom_filter_1_110Hz_notch_raw.fif"  %(subj,subj)
    #raw = mne.io.Raw(empty_room_raw_path)
    
    
    
    info['bads'] = []
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
    
    # create an ROI stc and save it
    if False:
        ROI_stc = np.ones([m,1])
        for i in range(n_ROI_valid):
            ROI_stc[ROI_list[i]] = i+2.0
        print np.max(ROI_stc)
        vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        stc = mne.SourceEstimate(data = ROI_stc,vertices = vertices, 
                                     tmin =0.0, tstep = 1.0 )
        stc.save(outpath+"ROI-stc")  
        print "ROI stc saved"
    
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
    
    if flag_time_smooth_source_noise or flag_space_smooth_source_noise:
        simu_path = "/home/ying/dropbox_unsync/MEEG_source_roi_cov_simu_and_data/"
        st_mat_name = simu_path + "source_noise_space_time_covar_chol_nndot%d.mat" % flag_nn_dot
        if not os.path.isfile(st_mat_name):
            # ======  create the spatial-temporal covariance matrix source noise
            space_covar = np.eye(m)*0.1
            a0 = 1.5  # exp (- a0 ||x-y||^2)
            for i0 in range(m):
                for i1 in range(m):
                    sign_factor =  np.dot(nn[i0,:], nn[i1,:]) if flag_nn_dot else 1.0
                    space_covar[i0,i1] += sign_factor* np.exp(-a0 * (np.sum((rr[i0,:]-rr[i1,:])**2)))
            space_covar /= np.max(space_covar)
            space_covar_chol = np.linalg.cholesky(space_covar)
    
            # temporal covariance
            time_covar =  np.eye(T+1)*0.1
            a = 1e-2
            for i in range(T+1):
                for j in range(T+1):
                    time_covar[i,j] += np.exp(- a*(i-j)**2)
            time_covar /= np.max(time_covar)
            time_covar_chol = np.linalg.cholesky(time_covar)
            st_mat_dict = dict(space_covar_chol = space_covar_chol,
                               time_covar_chol = time_covar_chol)                   
            scipy.io.savemat(st_mat_name, st_mat_dict)
        else:
            st_mat_dict = scipy.io.loadmat(st_mat_name)
            time_covar_chol, space_covar_chol = st_mat_dict['time_covar_chol'], st_mat_dict['space_covar_chol']
        
    if flag_space_smooth_source_noise:
        tmp_noise = np.dot(space_covar_chol, np.random.randn(q, m, T+1)).transpose([1,0,2])
    else:
        tmp_noise = np.random.randn(q, m, T+1)
    if flag_time_smooth_source_noise:
        J_noise = (tmp_noise).dot(time_covar_chol.T)
    else: 
        J_noise = (tmp_noise)
    
        # kronecker
        # covar  AXB,  first dim AA^T second B^T B
        # m by n, A = spatial_covar_chol, B = time_covar_chol
        #if flag_smooth_source_noise:
        #    tmp_noise = np.dot(space_covar_chol, np.random.randn(q, m, T+1)).transpose([1,0,2])
        #    J_noise = (tmp_noise).dot(time_covar_chol.T)
        #else:
        #    J_noise = np.random.randn(q,m,T+1)
    
    # if I want to use the smooth noise, I can set Sigma_J_list for different ROIs to be the same
    J = np.zeros([q,m,T+1])
    for i in range(n_ROI_valid):
        for j in range(len(ROI_list[i])):
            J[:,ROI_list[i][j],:] = L_list[i][j]*u[:,i,:] \
                  +J_noise[:, ROI_list[i][j],:]*np.sqrt(Sigma_J_list[i])
    if n_ROI_valid < n_ROI:
        i = -1
        J[:,ROI_list[i],:] = J_noise[:, ROI_list[i],:]*np.sqrt(Sigma_J_list[i])
  
    # generate J according to the model               
    #J = np.zeros([q,m,T+1])
    #for i in range(n_ROI_valid):
    #    for j in range(len(ROI_list[i])):
    #        J[:,ROI_list[i][j],:] = L_list[i][j]*u[:,i,:] \
    #              + np.random.randn(q, T+1)*np.sqrt(Sigma_J_list[i])
    #if n_ROI_valid < n_ROI:
    #    i = -1
    #    J[:,ROI_list[i],:] = np.random.randn(q, len(ROI_list[i]), T+1)*np.sqrt(Sigma_J_list[i])

    sel_cov = [l for l in range(len(noise_cov_reg.ch_names)) 
         if noise_cov_reg.ch_names[l] not in noise_cov_reg['bads']]
    Sigma_E = noise_cov_reg.data[:,sel_cov]
    # regularize the Sigma_E a bit 
    Sigma_E = Sigma_E[sel_cov,:]
    # regularize the cov matrix
    Sigma_E *= np.eye(Sigma_E.shape[0])*0.01 + np.ones(Sigma_E.shape)
    Sigma_E_chol = np.linalg.cholesky(Sigma_E)
    G = fwd['sol']['data'][sel_cov,:]
    n = G.shape[0]
    noise_M = np.dot(Sigma_E_chol, np.random.randn(q,n,(T+1)))
    M = (np.dot(G, J) + noise_M).transpose([1,0,2])
    
    # save the new regularized noise cov
    noise_cov_ch_names = [noise_cov_reg.ch_names[l] for l in sel_cov]
    noise_cov_new = mne.cov.Covariance(data = Sigma_E, names = noise_cov_ch_names,
                        bads = noise_cov_reg['bads'], projs = [], nfree = len(sel_cov),
                                        eig = None, eigvec = None, method = None)
    # save the noise_cov
    out_cov_fname = outpath + "-cov.fif"                                 
    noise_cov_new.save(out_cov_fname)
    
    # save an evoked data
    tmin, tstep = 0, 0.01
    vertices = src[0]
    vertices = [ src[0]['vertno'], src[1]['vertno']]
    tmp_stc = mne.SourceEstimate(J[0], vertices = vertices, 
                                     tmin = tmin, tstep = tstep)
    evoked = mne.simulation.simulate_evoked(fwd, tmp_stc, info, 
                                noise_cov_reg, snr = 1, iir_filter= None, verbose = False)
    evoked.info['bads'] = info['bads']
    evoked.save(outpath+"-ave.fif.gz")
   
    mat_dict = dict(J = J, M = M, u = u, G = G,
                fwd_path = fwd1_fname, noise_cov_path = out_cov_fname,
                ROI_list = ROI_list,
                A = A, Q = Q, Q0 = Q0,
                Sigma_J_list = Sigma_J_list,
                L_list = L_list,
                L_list_option = L_list_option,
                L_list_param = L_list_param if L_list_param is not None else 0.0,
                Sigma_E = Sigma_E)
    scipy.io.savemat(outpath+".mat", mat_dict, oned_as = "row")

 
        
        
        

    

