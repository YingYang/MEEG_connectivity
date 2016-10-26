# -*- coding: utf-8 -*-
import numpy as np
import sys
import scipy.io

import mne
from mne.forward import  is_fixed_orient, _to_fixed_ori
from mne.inverse_sparse.mxne_inverse import _prepare_gain
import copy

from multiprocessing import Pool

# no depth weithing is reapplied to the source solution

# for portability, I need to make write it as a package
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)

#=============================================================================
from ROI_Kalman_smoothing import EM, get_neg_llh_y, get_lsq_u, get_param_given_u
import ROI_cov as inst
def use_EM(ini_param):
    '''
    Utility function for parallel processing
    '''
    return EM(ini_param['y_array'],
              ini_param['G'],
              ini_param['ROI_list'],
              ini_param['Sigma_E'], 
              ini_param['Gamma0_0'],
              ini_param['A_0'], 
              ini_param['Gamma_0'],
              ini_param['sigma_J_list_0'],
              ini_param['L_list_0'],
              flag_A_time_vary = ini_param['flag_A_time_vary'],
              prior_Q0 = ini_param['prior_Q0'], 
              prior_A = ini_param['prior_A'], 
              prior_Q = ini_param['prior_Q'],
              prior_L_precision = ini_param['prior_L_precision'], 
              prior_sigma_J_list = ini_param['prior_sigma_J_list'],
              MaxIter0 = ini_param['MaxIter0'], 
              tol0 = ini_param['tol0'], 
              verbose0 = ini_param['verbose0'],
              MaxIter = ini_param['MaxIter'], 
              tol = ini_param['tol'], 
              verbose = ini_param['verbose'], 
              L_flag = ini_param['L_flag'])    

#==============================================================================
def get_estimate_ks(M, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name, 
                 prior_Q0 = None, prior_Q = None, prior_sigma_J_list = None, 
                 prior_A = None,
                 depth = None,
                 MaxIter0 = 100, MaxIter = 50, MaxIter_coarse = 10,
                 tol0 = 1E-4, tol = 1E-2,
                 verbose0 = True, verbose = False, verbose_coarse = True,
                 L_flag = False,
                 whiten_flag = True,
                 n_ini= 0, n_pool = 2, flag_A_time_vary = False, use_pool = False,
                 ini_Gamma0_list = None, ini_A_list = None, ini_Gamma_list = None,
                 ini_sigma_J_list = None, force_fixed=True, flag_inst_ini = True,
                 a_ini = 0.1):
    """
    Inputs: 
        M, [q, n_channels, n_times] sensor data
        ROI_list, ROI indices list
        fwd_path, full path of the forward solution
        evoked_path, full path of the evoked template
        noise_cov_path, full path of the noise covariance
        out_name, full path of the mat name to save
        
        # actually due to scale issues, no depth weighting should be allowed in the simulation. 
        # because normalizing G will result in strong violation of source generation assumptions
        priors:
        prior_Q0, prior_Q, prior_sigma_J_list, not implemented, may be inverse gamma or gamma
        prior_A, dict(lambda0 = 0.0, lambda1 = 1.0)
        
        depth: forward weighting parameter
        verbose:
        whiten_flag: if True, whiten the data, so that sensor error is identity  
        n_ini, number of random initializations
        
        # list of initial values,
        # ini_Gamma0_list, ini_A_list, ini_Gamma_list, ini_sigma_J_list must have the same length

    """
    if depth == None:
        depth = 0.0
    
    q,_,T0 = M.shape
    T = T0-1
    # this function returns a list, take the first element
    evoked = mne.read_evokeds(evoked_path)[0]
    # depth weighting, TO BE MODIFIED
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
    
    # prior for L
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
    prior_L_precision = copy.deepcopy(Q_L_list)
    for i in range(n_ROI_valid):
        prior_L_precision[i] = np.linalg.inv(Q_L_list[i]) 
    
    y_array = M.transpose([0,2,1]) # q,T,n    
    scale_factor = 1E-9
    p = n_ROI_valid
    
    L_list_0 = list()
    for i in range(n_ROI_valid):
        L_list_0.append(np.ones(ROI_list[i].size))
        
    # default param list, A being all zero
    
    ini_param_list = list()        
    Gamma0_0 = np.eye(p)*scale_factor
    Gamma_0 = np.eye(p)*scale_factor
    if flag_A_time_vary:
        A_0 = np.zeros([T,p,p])
        for t in range(T):
            A_0[t] = np.eye(p)*a_ini
    else:
        A_0 = np.eye(p)*a_ini
    sigma_J_list_0 = np.ones(p)*scale_factor
    
    if ini_Gamma0_list is None:
        ini_Gamma0_list = list() 
    if ini_A_list is None:
        ini_A_list = list() 
    if ini_Gamma_list is None:    
        ini_Gamma_list = list() 
    if ini_sigma_J_list is None:
        ini_sigma_J_list = list() 
    
    #if n_ini >= 0, append a new initialization, else do not
    if n_ini >= 0:
        ini_Gamma0_list.append(Gamma0_0)
        ini_A_list.append(A_0)
        ini_Gamma_list.append(Gamma_0)
        ini_sigma_J_list.append(sigma_J_list_0)
        ini_param_list = list()
        
    for l1 in range(len(ini_Gamma0_list)):
       ini_param_list.append(dict(y_array=y_array, G=G, ROI_list =ROI_list,
                        Sigma_E = Sigma_E, 
                        Gamma0_0 = ini_Gamma0_list[l1], A_0 = ini_A_list[l1], 
                        Gamma_0=  ini_Gamma_list[l1], sigma_J_list_0 = ini_sigma_J_list[l1], 
                        L_list_0 = L_list_0, flag_A_time_vary = flag_A_time_vary,
                        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
                        prior_L_precision = prior_L_precision, 
                        prior_sigma_J_list = prior_sigma_J_list,
                        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = False,
                        MaxIter = MaxIter_coarse, tol = tol, verbose = verbose_coarse, 
                        L_flag = L_flag))
    
    # second initialization, least squares
    m = G.shape[1]
    L = np.zeros([m, n_ROI_valid])
    for i in range(n_ROI_valid):
        L[ROI_list[i], i] = L_list_0[i]
    C = G.dot(L) 
    R0 = Sigma_E.copy() 
    for l in range(len(sigma_J_list_0)):
        R0 += sigma_J_list_0[l]**2 *  G[:, ROI_list[l]].dot(G[:, ROI_list[l]].T)      
    u_array_hat = get_lsq_u(y_array, R0,C)
    # set priors all to None,  avoid coordinate decent to get the global solution
    Gamma0_ls, A_ls, Gamma_ls = get_param_given_u(u_array_hat, Gamma0_0, A_0, Gamma_0, 
       flag_A_time_vary = flag_A_time_vary,
       prior_Q0 = None,  prior_A = None, prior_Q = None,
       MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0,
       MaxIter = MaxIter, tol = tol, verbose = verbose) 
    # debug 
    print "Gamma0_ls and Gamma_ls"
    print Gamma0_ls
    print Gamma_ls 
    
    if n_ini >= 0:      
        ini_param_list.append(dict(y_array=y_array, G=G, ROI_list =ROI_list,
                        Sigma_E = Sigma_E, 
                        Gamma0_0 = Gamma0_ls, A_0 = A_ls, Gamma_0= Gamma_ls, 
                        sigma_J_list_0 = sigma_J_list_0, L_list_0 = L_list_0,
                        flag_A_time_vary = flag_A_time_vary,
                        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
                        prior_L_precision = prior_L_precision, 
                        prior_sigma_J_list = prior_sigma_J_list,
                        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = False,
                        MaxIter = MaxIter_coarse, tol = tol, verbose = verbose_coarse, L_flag = L_flag))

    if flag_inst_ini: # run the instantaneous model to get initialization for Q and sigma_J_list
        print "initilization using my instantaneous model"        
        t_ind = 1
        MMT = M[:,:,t_ind].T.dot(M[:,:,t_ind])
        Qu0 = np.eye(p)*scale_factor**2
        Sigma_J_list0 = np.ones(len(ROI_list))*scale_factor**2
        # these parames are not used,
        alpha, beta = 1.0, 1.0; nu = p +1; V_inv = np.eye(p)*1E-4; eps = 1E-13;
        inv_Q_L_list = list()
        for i in range(n_ROI_valid):
            inv_Q_L_list.append(np.eye(len(ROI_list[i])))    
        Qu_hat0, Sigma_J_list_hat0, L_list_hat, obj = inst.get_map_coor_descent(
                            Qu0, Sigma_J_list0, L_list_0,
                          ROI_list, G, MMT, q, Sigma_E,
                          nu, V_inv, inv_Q_L_list, alpha, beta, 
                          prior_Q = False, prior_Sigma_J = False, prior_L = False ,
                          Q_flag = True, Sigma_J_flag = True, L_flag = False,
                          tau = 0.8, step_ini = 1.0, MaxIter = MaxIter, tol = tol,
                          eps = eps, verbose = verbose, verbose0 = verbose0, 
                          MaxIter0 = MaxIter0, tol0 = tol0)
        print Qu_hat0, Sigma_J_list_hat0
        ini_param_list.append(dict(y_array=y_array, G=G, ROI_list =ROI_list,
                        Sigma_E = Sigma_E, 
                        Gamma0_0 = np.linalg.cholesky(Qu_hat0), A_0 = A_0, 
                        Gamma_0 =  np.linalg.cholesky(Qu_hat0), 
                        sigma_J_list_0 = np.sqrt(Sigma_J_list_hat0),
                        L_list_0 = L_list_0, flag_A_time_vary = flag_A_time_vary,
                        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
                        prior_L_precision = prior_L_precision, 
                        prior_sigma_J_list = prior_sigma_J_list,
                        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = False,
                        MaxIter = MaxIter_coarse, tol = tol, verbose = verbose_coarse, 
                        L_flag = L_flag))
               
    if n_ini > 0 and flag_A_time_vary :
        # cut the time into n_ini segments evenly, compute the fixed A, and then concatenate them
        time_ind_dict_list = list() # each element is a dict, including l and time_ind_list_tmp
        for l in range(n_ini): # 1+2+..+ n_ini
            # segmant y_array!
            n_time_per_segment = (T+1)//(l+1)
            if l == 0:
                time_ind_dict_list.append(dict(l = l,time_ind =range(T+1)))
            else:
                for l0 in range(l):
                    time_ind =range(l0*n_time_per_segment, (l0+1)*n_time_per_segment+1)
                    time_ind_dict_list.append(dict(l = l, time_ind = time_ind))
                time_ind_dict_list.append(dict(l = l, time_ind =range((l0+1)*n_time_per_segment, T+1)))
        # Gamma0, Gamm0_0, L_list_0, sigma_J_list_0, are already defined
        tmp_A0 = np.eye(p)*0.9     
        ini_param_fixed_A_list = list()
        for l0 in range(len(time_ind_dict_list)):
            print l0
            y_array_tmp = y_array[:,time_ind_dict_list[l0]['time_ind'],:]
            print y_array_tmp.shape
            ini_param_fixed_A_list.append(dict(y_array=y_array_tmp, G=G, ROI_list =ROI_list,
                        Sigma_E = Sigma_E, 
                        Gamma0_0 = Gamma0_0, A_0 = tmp_A0, Gamma_0= Gamma_0, 
                        sigma_J_list_0 = sigma_J_list_0, L_list_0 = L_list_0,
                        flag_A_time_vary = False,
                        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
                        prior_L_precision = prior_L_precision, 
                        prior_sigma_J_list = prior_sigma_J_list,
                        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = False,
                        MaxIter = MaxIter_coarse, tol = tol, verbose = verbose_coarse, L_flag = L_flag))
        # solve the individual 
        if use_pool:
            pool = Pool(n_pool)
            result_fixed_list = pool.map(use_EM, ini_param_fixed_A_list)
            pool.close() 
        else:
            result_fixed_list = list()
            for l0 in range(len(ini_param_fixed_A_list)):
                print "fixed %d th ini_param" %l0
                result_fixed_list.append(use_EM(ini_param_fixed_A_list[l0]))
        # combine new A0_piecewise, add them to param list
        for l in range(n_ini):
            relevant_ind = [l0 for l0 in range(len(time_ind_dict_list))
                          if time_ind_dict_list[l0]['l'] == l]
            tmp_A0 = np.zeros([T,p,p])
            tmp_Q0 = np.zeros([p,p])
            for l0 in relevant_ind:
                tmp_time_ind = time_ind_dict_list[l0]['time_ind']
                for t0 in tmp_time_ind[1::]:
                    tmp_A0[t0-1,:,:] = result_fixed_list[l0]['A']
                tmp_Gamma = result_fixed_list[l0]['Gamma']
                tmp_Q0 += tmp_Gamma.dot(tmp_Gamma)
            tmp_Q0 /= np.float(len(relevant_ind))
            ini_param_list.append(dict(y_array=y_array, G=G, ROI_list =ROI_list,
                        Sigma_E = Sigma_E,  Gamma0_0 = Gamma0_0, A_0 = tmp_A0, 
                        Gamma_0= np.linalg.cholesky(tmp_Q0), 
                        sigma_J_list_0 = sigma_J_list_0, L_list_0 = L_list_0,
                        flag_A_time_vary = flag_A_time_vary,
                        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
                        prior_L_precision = prior_L_precision, 
                        prior_sigma_J_list = prior_sigma_J_list,
                        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = False,
                        MaxIter = MaxIter_coarse, tol = tol, verbose = False, L_flag = L_flag))
    # after obtaining the multiple starting points, solve them with a few iterations                           
    # try parallel processing
    print "optimizing %d initializations" % len(ini_param_list)
    if use_pool:
        print "using pool"
        pool = Pool(n_pool)
        result_list = pool.map(use_EM, ini_param_list)
        pool.close() 
    else:
        result_list = list()
        for l in range(len(ini_param_list)):
            result_list.append(use_EM(ini_param_list[l]))
 
    obj_all = np.zeros(len(result_list))
    for l in range(len(result_list)):
        obj_all[l] = result_list[l]['obj']
    print obj_all
    i_star = np.argmin(obj_all)
    
    ini_param = dict(y_array=y_array, G=G, ROI_list =ROI_list,
                        Sigma_E = Sigma_E, 
                        Gamma0_0 = result_list[i_star]['Gamma0'],
                        A_0 = result_list[i_star]['A'], 
                        Gamma_0=  result_list[i_star]['Gamma'], 
                        sigma_J_list_0 = result_list[i_star]['sigma_J_list'], 
                        L_list_0 = result_list[i_star]['L_list'],
                        flag_A_time_vary = flag_A_time_vary,
                        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
                        prior_L_precision = prior_L_precision, 
                        prior_sigma_J_list = prior_sigma_J_list,
                        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0,
                        MaxIter = MaxIter, tol = tol, verbose = verbose, L_flag = L_flag)

    result0 = use_EM(ini_param)  
    Gamma0_hat, A_hat, Gamma_hat  = result0['Gamma0'], result0['A'], result0['Gamma']
    sigma_J_list_hat   = result0['sigma_J_list']             
    L_list_hat = result0['L_list']
    print result0['obj']              
    result = dict(Q0_hat = Gamma0_hat.dot(Gamma0_hat.T),
                  Q_hat = Gamma_hat.dot(Gamma_hat.T),
                  A_hat = A_hat,
                  Sigma_J_list_hat = sigma_J_list_hat**2,
                  L_list_hat = L_list_hat,
                  u_array_hat = result0['u_t_T_array'], obj = result0['obj'])
    scipy.io.savemat(out_name, result)
        