# -*- coding: utf-8 -*-
import numpy as np
import sys
import scipy.io

import mne
from mne.forward import  is_fixed_orient, _to_fixed_ori
import copy

# no depth weithing is reapplied to the source solution

# for portability, I need to make write it as a package
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)

from ROI_cov import get_map_coor_descent, get_neg_llh
from ROI_cov_Kronecker import (get_mle_kron_cov,
                               get_neg_llh_kron,get_map_coor_descent_kron)                            

def get_estimate(filepath, outname, method = "ROIcov", 
                 loose = None, depth = None,
                 verbose = True, whiten_flag = True,
                 lambda2 = 1.0, Qu0 = None,
                 L_flag = True, L_list0 = None, Sigma_J_list0 = None):
    """
    filepath, full path of the file, with no ".mat" or "-ave.fif" suffix.
    e.g. filepath = "/home/ying/sandbox/MEG_simu_test"
    """
    
    
    if method == "ROIcovKronecker" and whiten_flag is False:
        raise ValueError("ROIcovKronecker requires pre-whitening")
    #========= load data ======================
    mat_dict = scipy.io.loadmat(filepath+".mat")
    ROI_list = list()
    n_ROI = len(mat_dict['ROI_list'][0])
    n_ROI_valid = n_ROI-1
    
    for i in range(n_ROI):
        ROI_list.append(mat_dict['ROI_list'][0,i][0])

    M = mat_dict['M']
    QUcov = mat_dict['QUcov']
    Tcov = mat_dict['Tcov']
    Sigma_J_list = mat_dict['Sigma_J_list'][0]
    L_list = list()
    for i in range(n_ROI_valid):
        L_list.append(mat_dict['L_list'][0,i][0])    
    
    fwd_path = mat_dict['fwd_path'][0]
    noise_cov_path = mat_dict['noise_cov_path'][0]
    Sigma_E = mat_dict['Sigma_E']
    G = mat_dict['G']
    
    # this function returns a list, take the first element
    evoked = mne.read_evokeds(filepath+"-ave.fif.gz")[0]
    # depth weighting has no effect, because this is force_fixed
    fwd = mne.read_forward_solution(fwd_path,force_fixed=True, surf_ori = True)
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
    
    # for some methods, only analyze a snapshot
    T = M.shape[-1]
    q = M.shape[0]
    t_ind = T//2+1
    m = rr.shape[0]
    
    all_ch_names = evoked.ch_names
    sel = [l for l in range(len(all_ch_names)) if all_ch_names[l] not in evoked.info['bads']]

    #======== ROI cov method
    if method in [ "ROIcov", "ROIcovKronecker"]:
        if loose is None and not is_fixed_orient(fwd):
            # follow the tf_mixed_norm
            fwd= copy.deepcopy(fwd)
            # it seems that this will result in different G from reading the forward with force_fixed = True
            _to_fixed_ori(fwd)
                
        if whiten_flag:
            Sigma_E_chol = np.linalg.cholesky(Sigma_E)
            Sigma_E_chol_inv = np.linalg.inv(Sigma_E_chol)
            G = np.dot(Sigma_E_chol_inv, G)
            # after whitening, the noise cov is assumed to identity
            #Sigma_E = (np.dot(Sigma_E_chol_inv, Sigma_E)).dot(Sigma_E_chol_inv.T)
            Sigma_E = np.eye(G.shape[0])
            M = (np.dot(Sigma_E_chol_inv, M)).transpose([1,0,2])

        # these arguments are going to be passed from inputs
        Q_flag, Sigma_J_flag, Tcov_flag  = True, True, True
        prior_Q, prior_Sigma_J, prior_Tcov = False, False, False
        prior_L = True if L_flag else False
        tau, step_ini, MaxIter, tol, MaxIter0, tol0, verbose0  = 0.8, 1.0, 10, 1E-5, 10, 1E-3, False
        # Create prior for L, not necessarily the same as the truth
        L_list_param = 1.5 # a exp (-b ||x-y||^2)
        Q_L_list = list()
        for i in range(n_ROI_valid):
            tmp_n = len(ROI_list[i])
            tmp = np.zeros([tmp_n, tmp_n])
            for i0 in range(tmp_n):
                for i1 in range(tmp_n):
                    tmp[i0,i1] = np.dot(nn[i0,:], nn[i1,:])* np.exp(-L_list_param * (np.sum((rr[i0,:]-rr[i1,:])**2)))
            #print np.linalg.cond(tmp)       
            Q_L_list.append(tmp)
        inv_Q_L_list = copy.deepcopy(Q_L_list)
        for i in range(n_ROI_valid):
            inv_Q_L_list[i] = np.linalg.inv(Q_L_list[i]) 
            
        
        n_trial, n_channel, T = M.shape
        # other priors, not used for now
        alpha, beta = 1.0, 1.0
        nu = n_ROI_valid +1
        V_inv = np.eye(n_ROI_valid)*1E-4
        eps = 1E-13
        
        V = np.eye(n_ROI_valid)
        nu1 = T+1
        V1 = np.eye(T)
        
        # how many times to randomly initialize the data
        if Qu0 is None:
            n_ini = 1
            Qu0 = np.eye(n_ROI_valid)*1E-18
        else:
            print "using given Qu0"
            n_ini = 1
            

        result_all = np.zeros(n_ini, dtype = np.object)
          
        # Tcov initialization to be added
        #Tcov0, _ = get_mle_kron_cov(M, tol = 1E-6, MaxIter = 100) 
        # a different initialization
        Tcov0 = np.zeros([T,T])
        for i in range(n_channel):
            Tcov0 += np.corrcoef(M[:,i,:].T)
        Tcov0 /= np.float(n_channel) 
        T00 = np.linalg.cholesky(Tcov0)
        
        for l in range(n_ini):
            if Sigma_J_list0 is None:
                "initializing sigma_J_list0"
                sigma_J_list0 = np.ones(n_ROI)*1E-18
                Sigma_J_list0 = sigma_J_list0**2
                
            if L_list0 is None:
                L_list0 = copy.deepcopy(L_list)
                for i in range(n_ROI_valid):
                    #L_list0[i] = np.random.randn(L_list0[i].size)
                    L_list0[i] = np.ones(L_list0[i].size)
                    print "L set to 1"
        
            if method is "ROIcov":
                # just analyze the middle time point.
                MMT = M[:,:,t_ind].T.dot(M[:,:,t_ind]) 
                if verbose:
                    print "initial obj"
                    Phi0 = np.linalg.cholesky(Qu0)       
                    sigma_J_list0 = np.sqrt(Sigma_J_list0)
                    obj0 = get_neg_llh(Phi0, sigma_J_list0, L_list0, 
                                       ROI_list, G, MMT, q, Sigma_E,
                                     nu, V_inv, inv_Q_L_list, alpha, beta,
                                     prior_Q, prior_Sigma_J, prior_L, eps)
                    print obj0                 
                    print "optimial obj" 
                    Phi = np.linalg.cholesky(QUcov) 
                    # lower case indicates the square root
                    sigma_J_list = np.sqrt(Sigma_J_list)                
                    obj_star = get_neg_llh(Phi, sigma_J_list, L_list, 
                                           ROI_list, G, MMT, q, Sigma_E,
                                     nu, V_inv, inv_Q_L_list, alpha, beta,
                                     prior_Q, prior_Sigma_J, prior_L, eps)  
                    print obj_star
                Qu_hat, Sigma_J_list_hat, L_list_hat, obj = get_map_coor_descent(
                        Qu0, Sigma_J_list0, L_list0,
                      ROI_list, G, MMT, q, Sigma_E,
                      nu, V_inv, inv_Q_L_list, alpha, beta, 
                      prior_Q, prior_Sigma_J, prior_L ,
                      Q_flag = Q_flag, Sigma_J_flag = Sigma_J_flag, L_flag = L_flag,
                      tau = tau, step_ini = step_ini, MaxIter = MaxIter, tol = tol,
                      eps = eps, verbose = verbose, verbose0 = verbose0, 
                      MaxIter0 = MaxIter0, tol0 = tol0)                       
             
                diag0 = np.sqrt(np.diag(Qu_hat))
                denom = np.outer(diag0, diag0)
                corr_hat = np.abs(Qu_hat/denom)
                
                result_all[l] = dict(obj = obj, Qu_hat = Qu_hat,
                                Sigma_J_list_hat = Sigma_J_list_hat,
                                L_list_hat = L_list_hat, corr_hat = corr_hat,
                                Tcov_hat = 0.0)                    
            elif method is "ROIcovKronecker":
                if verbose:
                    print "initial obj"
                    Phi0 = np.linalg.cholesky(Qu0)       
                    sigma_J_list0 = np.sqrt(Sigma_J_list0)
                    obj0 = get_neg_llh_kron(Phi0, sigma_J_list0, L_list0, T00, # unknown parameters
                                     ROI_list, G, M, q, 
                                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov) # prior flags
                    print obj0                 
                    print "optimial obj" 
                    Phi = np.linalg.cholesky(QUcov) 
                    sigma_J_list = np.sqrt(Sigma_J_list)
                    T0 = np.linalg.cholesky(Tcov)                
                    obj_star = get_neg_llh_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                                     ROI_list, G, M, q, 
                                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov) # prior flags 
                    print obj_star
                Qu_hat, Sigma_J_list_hat, L_list_hat, Tcov_hat, obj = get_map_coor_descent_kron(
                     Qu0, Sigma_J_list0, L_list0, Tcov0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                     Q_flag = Q_flag, Sigma_J_flag = Sigma_J_flag, L_flag = L_flag, Tcov_flag = Tcov_flag,
                     tau = tau, step_ini = step_ini, MaxIter = MaxIter, tol = tol, verbose = verbose, # optimization params
                     MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0)
                diag0 = np.sqrt(np.diag(Qu_hat))
                denom = np.outer(diag0, diag0)
                corr_hat = np.abs(Qu_hat/denom)
                result_all[l] = dict(obj = obj, Qu_hat = Qu_hat, Tcov_hat = Tcov_hat,
                                Sigma_J_list_hat = Sigma_J_list_hat,
                                L_list_hat = L_list_hat, corr_hat = corr_hat, 
                                method = method, lambda2 = 0.0)
        # choose the best results
        obj_list = np.zeros(n_ini)
        for l in range(n_ini):
            obj_list[l] = result_all[l]['obj']
        result = result_all[np.argmin(obj_list)]
    
    # can do dSPM too,  directly apply the kernel
    elif method in ["mneFlip","mneTrueL","mnePairwise",
                    "mneTrueLKronecker", "mneFlipKronecker"]:
        mne_method = "MNE" # can be MNE        
        q = M.shape[0]
        # create the inverse operator
        inv_op = mne.minimum_norm.make_inverse_operator(evoked.info, fwd,
                        noise_cov, loose = loose, depth = depth, fixed = True)
        # apply the inverse
        # create an epoch object with the data
        # the events parameter here is fake.
        M_aug = np.zeros([q, evoked.info['nchan'], T])
        M_aug[:,sel,:] = M.copy()
        epochs = mne.EpochsArray(data = M_aug, info = evoked.info,
                                 events = np.ones([q,3], dtype = np.int),
                    tmin = evoked.times[0], event_id = None, reject = None)
        source_sol = mne.minimum_norm.apply_inverse_epochs(epochs, 
              inv_op, lambda2 = lambda2, method = mne_method, 
              nave = 1)
        m, T = source_sol[0].data.shape
        J_two_step = np.zeros([q,m,T])
        for r in range(q):
            J_two_step[r] = source_sol[r].data
        
        if method in ["mneFlip","mneTrueL",
                      "mneTrueLKronecker", "mneFlipKronecker"]:
            U_two_step = np.zeros([q,n_ROI_valid,T])
            for i in range(n_ROI_valid):
                J_tmp = J_two_step[:,ROI_list[i],:]
                if method in ["mneFlip","mneFlipKronecker"]:
                    tmp_nn = nn[ROI_list[i],:]
                    tmpu, tmpd, tmpv = np.linalg.svd(tmp_nn)
                    tmp_sign = np.sign(np.dot(tmp_nn, tmpv[0]))
                    U_two_step[:,i,:] = np.mean(J_tmp.transpose([0,2,1])*tmp_sign, axis = -1)
                elif method in ["mneTrueL", "mneTrueLKronecker"]:
                    # U = (L^T L)^{-1} L^T J
                    tmp_true_L = L_list[i]
                    tmp_LTL = np.dot(tmp_true_L, tmp_true_L)
                    U_two_step[:,i,:] = (np.dot(J_tmp.transpose([0,2,1]), tmp_true_L)/tmp_LTL)
            if method in ["mneFlip","mneTrueL"]:
                U_two_step0 = U_two_step[:,:,t_ind]
                Qu_hat = np.cov(U_two_step0.T)
                Tcov_hat = 0.0
            else:
                # kronecker
                Tcov_hat, Qu_hat = get_mle_kron_cov(U_two_step, tol = 1E-6, MaxIter = 100)
            
            diag0 = np.sqrt(np.diag(Qu_hat))
            denom = np.outer(diag0, diag0)
            corr_hat = np.abs(Qu_hat/denom)
        elif method in ["mnePairwise"]:
            Qu_hat, Tcov_hat = 0.0,0.0
            corr_hat = np.eye(n_ROI_valid) 
            for l1 in range(n_ROI_valid):
                for l2 in range(l1+1, n_ROI_valid):
                    J_tmp1 = J_two_step[:,ROI_list[l1], t_ind].T
                    J_tmp2 = J_two_step[:,ROI_list[l2], t_ind].T
                    tmp_corr = np.corrcoef(np.vstack([J_tmp1, J_tmp2]))
                    tmp_corr_valid = tmp_corr[0:J_tmp1.shape[0], J_tmp1.shape[0]::]
                    corr_hat[l1,l2] = np.mean(np.abs(tmp_corr_valid))
                    corr_hat[l2,l1] = corr_hat[l1,l2]
        result = dict(obj = 0.0, Qu_hat = Qu_hat, Tcov_hat = Tcov_hat,
                                Sigma_J_list_hat = 0.0, 
                                L_list = 0.0, corr_hat = corr_hat,
                                method = method, lambda2 = lambda2)
    # save the result
    scipy.io.savemat(outname, result)
        