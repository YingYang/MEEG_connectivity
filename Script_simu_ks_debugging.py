# -*- coding: utf-8 -*-
"""
    #fwd_path = mat_dict['fwd_path'][0]
    #print fwd_path
    #fwd = mne.read_forward_solution(fwd_path, force_fixed=True, surf_ori = True)
    #print np.linalg.norm(mat_dict['G'] - fwd['sol']['data'])
    u_array = mat_dict['u'].transpose([0,2,1])
    
    # debug, once the free-orentation fwd is saved and reloaded, G changes to weird values
    # Also, copying the file within python did not help. 
    #subj = "Subj1"
    #fwd_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/fwd/" \
    #                + "%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
    # keep fixed orientation                 
    #fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
    #mne.write_forward_solution(outpath+"-fwd.fif", fwd, overwrite = True) 
    # debug, why after saving, the same forward has totally different G
    #fwd1 = mne.read_forward_solution(outpath+"-fwd.fif", force_fixed=True, surf_ori=True )
    #print np.linalg.norm( mat_dict['G']- fwd1['sol']['data'])
"""
import numpy as np
import mne
import sys
import scipy.io
import copy
import os

# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)         
import matplotlib
#matplotlib.use('Agg')              
import matplotlib.pyplot as plt
#from get_simu_data_ks import get_simu_data_ks
from get_estimate_baseline import get_estimate_baseline 
from get_estimate_ks import get_estimate_ks 
from ROI_Kalman_smoothing import get_param_given_u


#%% create simulations
#=============================================================================
simu_path = "/home/ying/dropbox_unsync/MEEG_source_roi_cov_simu_and_data/"
simu_id_start = 6
simu_id_end = 5
simu_id = range(simu_id_start, simu_id_end)
n_simu = len(simu_id)
T = 20
q = 200

#======== use freesurfer anatomical labels
anat_ROI_names= ['pericalcarine-lh', 'pericalcarine-rh',
                 #'lateraloccipital-lh', 'lateraloccipital-rh',
                'parahippocampal-lh', 'parahippocampal-rh',]
                 #'medialorbitofrontal-lh','medialorbitofrontal-rh']

subjects_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
subj = "Subj1"
labels0 = mne.read_labels_from_annot(subj, parc='aparc',
                                    subjects_dir=subjects_dir)
labels = list()
for i in range(len(anat_ROI_names)//2):
    tmp_label_list = list()
    for j in range(2):
        tmp_label = [label for label in labels0 if label.name in anat_ROI_names[2*i+j:2*i+j+1]]
        tmp_label_list.append(tmp_label[0])
    labels.append(tmp_label_list[0] + tmp_label_list[1])
p = len(labels)  
                    
#======== or use my hand-drawn labels, TBA  
#n_alpha = len(alpha_list);
# also define the gamma_distribution of sigma_i^2
flag_random_A = False
flag_smooth_source_noise = True
space_smooth = False
time_smooth = True
flag_nn_dot = False

        

# sensor/source noise ratio is also controled by alpha and the scale factor
if True:    
    alpha = 5.0
    morlets = mne.time_frequency.morlet(sfreq = 10, freqs = [9,10], sigma = 5 )
    f01 = np.real(morlets[1])[0:T]
    f01 /= f01.max()
    f10 = np.real(morlets[1][3:T+3])
    f10 /= f10.max()
    A = np.zeros([T,p,p])
    for t in range(T):
        A[t] = np.eye(p)*0.5
    # feed-back
    A[:,0,1] = f01*np.random.rand(1)*np.sign(np.random.randn(1))
    # feed-fwd
    A[:,1,0] = f10*np.random.rand(1)*np.sign(np.random.randn(1))
    
    print np.max(np.abs(A),axis = 0)
            
    tmp = np.random.randn(p,p)
    r = np.random.gamma(shape=0.5, scale=1.0, size=p)
    Q = np.dot(tmp*r, (tmp*r).T)
    Q += np.eye(p)
    diag = np.sqrt(np.diag(Q))
    denom = np.outer(diag, diag)
    Q = Q/denom* alpha

    tmp = np.random.randn(p,p)
    r = np.random.gamma(shape=0.5, scale=1.0, size=p)
    Q0 = np.dot(tmp*r, (tmp*r).T)
    Q0 += np.eye(p)
    diag = np.sqrt(np.diag(Q0))
    denom = np.outer(diag, diag)
    Q0 = Q0/denom* alpha
    
    scale_factor = 1E-9
    Q = Q*scale_factor**2
    Q0 =Q0*scale_factor**2
    
    #========Sigma_J_list ===
    #x = np.arange(0,5,0.01)
    #plt.plot(x,scipy.stats.gamma.pdf(x,2,0,1))
    #Sigma_J_list = np.random.gamma(shape=2, scale=1.0, size= p+1)
    Sigma_J_list = np.ones(p+1)*scale_factor**2
    
    #======== 
    outpath = simu_path + \
       "/%s_ROI_alpha%1.1f_simu%d_randA%d_smthns%d_space%d_tmp%d" \
        %(p,alpha, 6, flag_random_A, flag_smooth_source_noise, space_smooth, time_smooth)
    #get_simu_data_ks_ico_4(q,T, labels, outpath,
    #                  A, Q, Q0, Sigma_J_list, 
    #                  L_list_option = 0, L_list_param = None,
    #                  subj = subj, flag_smooth_source_noise 
    #                  = flag_smooth_source_noise)
                      
                      
                      
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
    cov_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/" \
                    + "%s_STFT-R_MEGnoise_cov-cov.fif" %(subj)
    noise_cov = mne.read_cov(cov_fname) 

    # obtain evoked info
    evoked_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"+\
                "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica-epo.fif.gz" %(subj,subj)
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

    p = Q.shape[0]
    if p != n_ROI_valid or len(Sigma_J_list)-1!= p :
        raise ValueError("covariance size does not match ROI sets")
    
    # ==========generate L===============
    L_list = np.zeros(n_ROI_valid, dtype = np.object)
    # L_list fixed to 1
    for i in range(n_ROI_valid):
        tmp = np.ones( [len(ROI_list[i])])
        L_list[i] = tmp 

    # generate U
    u = np.zeros([q, p, T+1])
    for r in range(q):
        u[r,:,0] = np.random.multivariate_normal(np.zeros(p), Q0)
        for t in range(1,T+1):
            tmpA = A[t-1].copy() 
            u[r,:,t] = np.random.multivariate_normal(np.zeros(p), Q) + tmpA.dot(u[r,:,t-1])
    
    diag_val = 0.1
    # ======  create the spatial-temporal covariance matrix source noise
    if space_smooth:
        space_covar = np.eye(m)*diag_val
        a0 = 1.5  # exp (- a0 ||x-y||^2)
        for i0 in range(m):
            for i1 in range(m):
                tmp_factor = np.dot(nn[i0,:],nn[i1,:]) if flag_nn_dot else 1.0
                space_covar[i0,i1] += tmp_factor *np.exp(-a0 * (np.sum((rr[i0,:]-rr[i1,:])**2)))
        space_covar /= np.max(space_covar)
        space_covar_chol = np.linalg.cholesky(space_covar)

    # temporal covariance
    if time_smooth:
        time_covar =  np.eye(T+1)*0.1
        a = 1e-2
        for i in range(T+1):
            for j in range(T+1):
                time_covar[i,j] += np.exp(- a*(i-j)**2)
        time_covar /= np.max(time_covar)
        time_covar_chol = np.linalg.cholesky(time_covar)

   
    # debug:
    #space_cov = space_covar_chol.dot(space_covar_chol.T)
    # plot the space cov within each ROI
    #for i in range(len(ROI_list)):
    #    plt.figure(); plt.imshow((space_cov[ROI_list[i],:])[:, ROI_list[i]], interpolation = "none"); plt.colorbar();
    # kronecker
    # covar  AXB,  first dim AA^T second B^T B
    # m by n, A = spatial_covar_chol, B = time_covar_chol.T
    # try only add temporal no spatial
    #space_covar_chol = np.eye(m)
    if space_smooth:
        tmp_noise = np.dot(space_covar_chol, np.random.randn(q, m, T+1)).transpose([1,0,2])
    else:
        tmp_noise = np.random.randn(q, m, T+1)
    if time_smooth:
        J_noise = (tmp_noise).dot(time_covar_chol.T)
    else: 
        J_noise = (tmp_noise)
    
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
                Sigma_E = Sigma_E)
    scipy.io.savemat(outpath+".mat", mat_dict, oned_as = "row")
    print "simulation saved"

#%% solving the simulations, using mne and ks
# define some function and call them is easier!!
#outpath = "/home/ying/dropbox_unsync/MEEG_source_roi_cov_simu_and_data/2_ROI_alpha10.0_simu1_randA0_smthns1"
#def get_solution(outpath, lambda2_seq):                    
    #=======solution========================= 
if False:
    mat_dict = scipy.io.loadmat(outpath)
    ROI_list = list()
    n_ROI = len(mat_dict['ROI_list'][0])
    n_ROI_valid = n_ROI-1
    for i in range(n_ROI):
        ROI_list.append(mat_dict['ROI_list'][0,i][0])
    M = mat_dict['M']

    fwd_path = mat_dict['fwd_path'][0]
    noise_cov_path = mat_dict['noise_cov_path'][0]
    evoked_path = outpath+"-ave.fif.gz" 
    
    prior_A = dict(lambda0 = 0.0, lambda1 = 0.1)
    #prior_A = None
    prior_Q0, prior_Q, prior_sigma_J_list = None,None,None    
    MaxIter0, MaxIter = 100, 40
    tol0,tol = 1E-4, 2E-2
    verbose0, verbose = False, True
    L_flag = False
    whiten_flag = True
    depth=None
    flag_A_time_vary = True
    
    lbdid = 0
    tmp_lambda2 = 1.0
    for flag_sign_flip in [ False]:  
        out_name_mne = outpath + "_mne_sol_lbdid%d_flip%d" %(lbdid, flag_sign_flip)
        get_estimate_baseline(M, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_mne, 
                 method = "MNE", lambda2 = tmp_lambda2, prior_Q0 = prior_Q0, 
                 prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                 prior_A = prior_A, depth = depth, MaxIter0 = MaxIter0, 
                 MaxIter = MaxIter, tol0 = tol0, tol = tol, verbose0 = verbose, 
                 verbose =verbose, flag_A_time_vary = flag_A_time_vary, 
                 flag_sign_flip = flag_sign_flip) 
    
    tmp_out_name_mne = outpath + "_mne_sol_lbdid%d_flip%d" %(lbdid, False) 
    result_mne = scipy.io.loadmat(tmp_out_name_mne)
    
    ini_Gamma0_list = [np.linalg.cholesky(result_mne['Q0_hat'])]
    ini_A_list = [result_mne['A_hat']] #
    ini_Gamma_list = [np.linalg.cholesky(result_mne['Q_hat'])]
    ini_sigma_J_list = [np.sqrt(result_mne['Sigma_J_list_hat'][0])]
    print result_mne['Q0_hat'], result_mne['Q_hat'], result_mne['Sigma_J_list_hat'][0]

    # debugging
    if True:
        ini_Gamma0_list = [np.linalg.cholesky(mat_dict['Q0']), 
                           np.linalg.cholesky(result_mne['Q0_hat'])]
        ini_A_list = [mat_dict['A'], 
                      result_mne['A_hat']] #
        ini_Gamma_list = [np.linalg.cholesky(mat_dict['Q']), 
                          np.linalg.cholesky(result_mne['Q_hat'])]
        ini_sigma_J_list = [np.sqrt(mat_dict['Sigma_J_list'][0]),
                            np.sqrt(result_mne['Sigma_J_list_hat'][0])] 
        print ini_sigma_J_list
        
    #del(result_mne)
    
    out_name_ks = outpath + "_ks_sol"  
    if True:        
        get_estimate_ks(M, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_ks, 
                     prior_Q0 = prior_Q0, prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                     prior_A = prior_A,
                     depth = depth, MaxIter0 = MaxIter0, MaxIter = MaxIter,
                     tol0 = tol0, tol = tol,
                     verbose0 = verbose0, verbose = verbose, verbose_coarse = True,
                     L_flag = L_flag, whiten_flag = whiten_flag, n_ini = -1, 
                     flag_A_time_vary = flag_A_time_vary, use_pool = False, 
                     MaxIter_coarse = 2, ini_Gamma0_list = ini_Gamma0_list,
                     ini_A_list = ini_A_list, ini_Gamma_list = ini_Gamma_list,
                     ini_sigma_J_list = ini_sigma_J_list, flag_inst_ini = False)
    
    # evaluation
    flag_empi_true = True
    Q = mat_dict['Q']
    #Q0 = mat_dict['Q0']
    A = mat_dict['A']
    
    #Sigma_J_list = mat_dict['Sigma_J_list'][0]
    #L_list = list()
    #for i in range(n_ROI_valid):
    #    L_list.append(mat_dict['L_list'][0,i][0])
    u_array = mat_dict['u'].transpose([0,2,1])
    
    J = mat_dict['J']
    u_array_from_J = np.zeros([q,T+1,p])
    for r in range(q):
        for i in range(p):
            u_array_from_J[r,:,i] = np.mean(J[r, ROI_list[i],:], axis = 0)
    # try to estimate A and Q from u_array  
    p = Q.shape[0]
    Gamma0_0 = np.eye(p)
    T = A.shape[0]        
    A_0 = np.zeros([T,p,p])
    Gamma_0 = np.eye(p)
    # first run the non_prior version to get a global solution
    Gamma0_1, A_1, Gamma_1 = get_param_given_u(u_array_from_J.copy(), 
                                               Gamma0_0, A_0, Gamma_0, 
       flag_A_time_vary = True,
       prior_Q0 = None, prior_A = None, prior_Q = None,
       MaxIter0 = 100, tol0 = 1E-4, verbose0 = False,
       MaxIter = 100, tol = 1E-4, verbose = False)
   
    if flag_empi_true:   
        A_true = A_1.copy()
        Q_true = Gamma_1.dot(Gamma_1.T) 
        u_array_true = u_array_from_J.copy()
    else:
        A_true = A.copy()
        Q_true = Q.copy()
        u_array_true = u_array.copy()
        
    tmp_Q_corr = np.sqrt(np.diag(Q_true))
    Q_abs_corr = np.abs(Q_true/np.outer(tmp_Q_corr, tmp_Q_corr))
          
    sol_out_names = [outpath + "_ks_sol"]
    for l0 in [lbdid]:
        for flag_sign_flip in [False]:  
            sol_out_names.append(outpath + "_mne_sol_lbdid%d_flip%d" %(l0, flag_sign_flip))
    
    print sol_out_names
    n_sol = len(sol_out_names)
    A_error = np.zeros(n_sol)  # raw relative error
    A_abs_error = np.zeros(n_sol) # relative error of absolute value of A
    Q_error = np.zeros(n_sol)  # raw relative error 
    Q_abs_error = np.zeros(n_sol) # relative error of absolute value of correlation
    u_corr = np.zeros(n_sol)
    u_error = np.zeros(n_sol)

    result_list = list()
    for l0 in range(len(sol_out_names)):
        result = scipy.io.loadmat(sol_out_names[l0])
        result_list.append(result)
        A_hat, Q_hat, u_array_hat = result['A_hat'], result['Q_hat'], result['u_array_hat']
        
        A_error[l0] = np.sqrt(np.sum( (A_hat-A_true)**2 ))/np.sqrt( np.sum(A_true**2) )
        A_abs_error[l0] = np.sqrt(np.sum( (np.abs(A_hat)- np.abs(A_true) )**2 ))/np.sqrt(np.sum(np.abs(A_true)**2))
        Q_error[l0] = np.sqrt(np.sum( (Q_hat-Q_true)**2) )/np.sqrt(np.sum(Q_true**2))
        
        tmp = np.sqrt(np.diag(Q_hat))
        Q_abs_corr_hat = np.abs(Q_hat/np.outer(tmp, tmp))
        Q_abs_error[l0] = np.sqrt(np.sum( (Q_abs_corr_hat-Q_abs_corr)**2 ))/np.sqrt(np.sum(Q_abs_corr**2))
        
        #q = M.shape[0]
        corr_ts = np.zeros(p)
        # compute the correlation fo the time coureses
        #for r in range(q):
        for i in range(p):
            corr_ts[i] = np.abs(np.corrcoef(u_array_hat[:,:,i].ravel(), u_array[:,:,i].ravel())[0,1])
            #plt.figure(); plt.plot(u_array_hat[:,:,i].ravel(), u_array[:,:,i].ravel(), '.')
        u_corr[l0] = corr_ts.mean()
        u_error[l0] = np.sqrt(np.sum( (u_array_hat-u_array)**2) )/ np.sqrt(np.sum(u_array**2))
 
    print "A_error"
    print  A_error
    print "u_corr"
    print  u_corr
    print "Q_error"
    print  Q_error
    print "u_error"
    print  u_error
    
    plt.figure()
    trial_id = 0
    col_seq = ['r','g','b']
    count = 0
    roi_id = 0
    for u in [u_array_true, result_list[0]['u_array_hat'], result_list[1]['u_array_hat']]:
        plt.plot(u[trial_id,:,roi_id], col_seq[count]);
        count += 1
    plt.legend(['truth', 'ks','mne'])
    
    # plot the correlation with u truth
    plt.figure(figsize = (6,4))
    count = 0
    for l0 in range(len(result_list)):
        u_array_hat = result_list[l0]['u_array_hat']
        for i in range(p):
            count += 1
            plt.subplot(2,p,count)
            plt.plot(u_array_hat[:,:,i].ravel(), u_array[:,:,i].ravel(), '.')    
            plt.title('%d_th ROI' % i)
    plt.tight_layout()
    
    # plot the true A
    names = ['truth','ss','mne']
    plt.figure(figsize =(6,3))
    count = 0
    prop = {'size': 10}
    for i0 in range(p):
        for i1 in range(p):
            plt.subplot(p,p, count+1)
            _ = plt.plot(A[:,i0,i1]);
            _ = plt.plot(result_list[0]['A_hat'][:,i0,i1])
            _ = plt.plot(result_list[1]['A_hat'][:,i0,i1])
            _ = plt.title('A[:,%d,%d]'%(i0+1,i1+1))
            if count == 0:
                _ = plt.legend(names, loc = 8, ncol = 2, prop = prop )
                _ = plt.xlabel('time index')                   
                
            _ = plt.ylim(-1,1)
            count += 1
    plt.tight_layout(0.01)
   
   
    # debug: visualize the spatial-temporal cov of noise_J
    plt.figure(); plt.imshow(np.cov(J_noise[:,:,0].T), interpolation = "none"); plt.colorbar()
    plt.figure(); plt.imshow(np.cov(J_noise[:,0,:].T), interpolation = "none"); plt.colorbar()
    
    
        

