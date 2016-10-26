# -*- coding: utf-8 -*-

"""
In the simulation and Kalman filter E-step, Q may vary. 
But Q is always fixed in the actual EM model fitting. 
"""
import numpy as np
import copy
import scipy.io
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)

from ROI_Kalman_smoothing import * 
from get_estimate_ks import use_EM
import matplotlib.pyplot as plt  
#if False:
 
if False:
    """ test the prior objectives and gradients"""
    step = 1E-5
    maxiter = 1000
    alpha,beta, flag_inv = 2.0,3.0, False
    prior_sigma_J_list = dict(alpha=alpha, beta = beta, flag_inv = flag_inv)
    # test gamma/inverse gamma priors
    sigma_J_list = np.random.gamma(shape = alpha, scale = 1.0/beta, size = 5)
    for l0 in range(maxiter):
        print obj_prior_sigma_J_list(sigma_J_list, prior_sigma_J_list)
        #print sigma_J_list**2
        sigma_J_list -= step*grad_prior_sigma_J_list(sigma_J_list, prior_sigma_J_list)            
    # mode of Gamma (alpha-1)/beta, mode of inv Gamma beta/(alpha+1)    
    
    # to be further checked
    p, nu = 3,10
    V = np.eye(p)
    V[1,2] = V[2,1] = 0.5
    prior_Q = dict(nu = nu, V = V, flag_inv = flag_inv)
    tmp = np.random.randn(p,p)
    Gamma = tmp*0 + np.linalg.cholesky(V)
    for l0 in range(maxiter):
        print obj_prior_Q(Gamma, prior_Q)
        Gamma -= step*grad_prior_Q(Gamma, prior_Q)  
    # mode of Wishart is (n-p-1)* invV^{-1}, mode of inv Wishart is V/(nu+p+1)
    print Gamma
    print np.linalg.cholesky(V/(nu+p+1))
    print np.linalg.cholesky((nu-p-1)* np.linalg.inv(V))

if True: 
    flag_use_simulation = False
    if flag_use_simulation:
        p = 2
        #=============use the generated simulation data
        outpath = "/home/ying/Dropbox/tmp/ROI_cov_simu/%d_ROIs_ks_simu0" %(p)
        mat_dict = scipy.io.loadmat(outpath)
        ROI_list = list()
        n_ROI = len(mat_dict['ROI_list'][0])
        n_ROI_valid = n_ROI-1
        for i in range(n_ROI):
            ROI_list.append(mat_dict['ROI_list'][0,i][0])
        M = mat_dict['M']
        T = M.shape[2]-1
        Q = mat_dict['Q']
        Q0 = mat_dict['Q0']
        A = mat_dict['A']
        Sigma_J_list = mat_dict['Sigma_J_list'][0]
        L_list = list()
        for i in range(n_ROI_valid):
            L_list.append(mat_dict['L_list'][0,i][0]) 
        fwd_path = mat_dict['fwd_path'][0]
        noise_cov_path = mat_dict['noise_cov_path'][0]
        evoked_path = outpath+"-ave.fif.gz"
        G = mat_dict['G']
        Sigma_E = mat_dict['Sigma_E']
        n,m = G.shape
        sigma_J_list = np.sqrt(mat_dict['Sigma_J_list'][0])
        q,T = M.shape[0], M.shape[2]-1
        scale_factor = 1E-9
        
        # debug
        #print "create new Sigma_E"
        #tmp = np.random.randn(n,n)*np.random.randn(n)
        #Sigma_E = tmp.dot(tmp.T)/n
        #Sigma_E = Sigma_E*1E-18
        

    else:
        np.random.RandomState(None)
        #m,n,q,T = 5000,306,100,20 
        m,n,q,T = 500,20,100,20 
        np.random.RandomState()
        r = np.int(np.floor(n*1.0))
        G = (np.random.randn(n,r)* np.random.rand(r)).dot(np.random.randn(r,m))
        #G = np.random.randn(n,m)
        normalize_G_flag = False
        if normalize_G_flag:
            G /= np.sqrt(np.sum(G**2,axis = 0)) 
    
        n_ROI = 10
        #n_ROI_valid = n_ROI-1
        n_ROI_valid = n_ROI
        ROI_list = list()
        n_dipoles = m//n_ROI
        for l in range(n_ROI-1):
            ROI_list.append( np.arange(l*n_dipoles,(l+1)*n_dipoles))
        ROI_list.append( np.arange((n_ROI-1)*n_dipoles,m))
    
    
        alpha = 1.0
        p = n_ROI_valid
        tmp = np.random.randn(p,p)
        r = np.random.gamma(shape=0.5, scale=1.0, size=p)
        Q = np.dot(tmp*r, (tmp*r).T)
        Q += np.eye(p)
        diag = np.sqrt(np.diag(Q))
        denom = np.outer(diag, diag)
        Q = Q/denom* alpha
        tmp = np.random.randn(p,p)
        Q0 = np.dot(tmp*r, (tmp*r).T) + np.eye(p)
        
        scale_factor = 1.0
        Q = Q*scale_factor**2
        Q0 =Q0*scale_factor**2
        
        sigma_J_list = np.ones(n_ROI)*0.1
        sigma_J_list[-1] = 0.5
        
        p = n_ROI_valid
        A = np.zeros([T,p,p])
        # create a full feed-forward and feedback loop
        for t in range(T):
            A[t] = np.eye(p)*0.5
        A[:,0,2] = 0.8*np.sin(np.arange(0,T))
        A[:,1,3] = 0.8*np.cos(np.arange(0,T))
        tmp = np.random.randn(n,n)*np.random.randn(n)
        Sigma_E = tmp.dot(tmp.T)/n
    
    if False:
        a0,b0 = 1.0, 1E-5 # a exp (-b ||x-y||^2)
        prior_L_precision = list()
        for i in range(n_ROI_valid):
            tmp_n = len(ROI_list[i])
            tmp = np.zeros([tmp_n, tmp_n])
            for i0 in range(tmp_n):
                for i1 in range(tmp_n):
                    tmp[i0,i1] = a0 * np.exp(-b0 * np.sum((G[:,i0]-G[:,i1])**2))
            prior_L_precision.append(np.linalg.inv(tmp))
    prior_L_precision = None
    
    print "nmqT"
    print n,m,q,T
    
        
    L_list = list()
    for i in range(n_ROI_valid):
        tmp_n = len(ROI_list[i])
        #tmp = np.random.multivariate_normal(np.zeros(tmp_n), np.linalg.inv(prior_L_precision[i]))
        tmp = np.ones(tmp_n)        
        L_list.append(tmp)     
   
    L = np.zeros([m, n_ROI_valid])
    for i in range(n_ROI_valid):
        L[ROI_list[i], i] = L_list[i]
    

    C = G.dot(L)
    R = Sigma_E.copy() 
    for l in range(len(sigma_J_list)):
        R += sigma_J_list[l]**2 *  G[:, ROI_list[l]].dot(G[:, ROI_list[l]].T) 
    #R = (R+R.T)/2.0
    #print np.linalg.cholesky(R)
    
    # weird, why will R become singular # G Q_J GT should be full rank!!!
    # debug
    '''
    R1 = np.zeros([n,n])
    for l in range(len(sigma_J_list)):
        R1 += sigma_J_list[l]**2 *  G[:, ROI_list[l]].dot(G[:, ROI_list[l]].T) 
    Q_J = np.zeros(m)
    for l in range(len(sigma_J_list)):
        Q_J[ROI_list[l]] = sigma_J_list[l]**2 
    R2 = reduce(np.dot, [G, np.diag(Q_J), G.T])
    '''
    
    time_covar =  np.eye(T+1)*0.1
    a = 1e-2
    for i in range(T+1):
        for j in range(T+1):
            time_covar[i,j] += np.exp(- a*(i-j)**2)
    time_covar /= np.max(time_covar)
    time_covar_chol = np.linalg.cholesky(time_covar)
    
    # whether to directly simulate the data
    flag_simu_data_dir = False
    if flag_simu_data_dir:
        u_array, y_array = simulate_kalman_filter_data(T, Q0, A, Q, C, R, q)
    else:
        u = np.zeros([q, p, T+1])
        for r in range(q):
            u[r,:,0] = np.random.multivariate_normal(np.zeros(p), Q0)
            for t in range(1,T+1):
                tmpA = A[t-1].copy() 
                u[r,:,t] = np.random.multivariate_normal(np.zeros(p), Q) + tmpA.dot(u[r,:,t-1])    
        
        Flag_time_smooth = True
        if Flag_time_smooth:
            tmp_noise = np.random.randn(q, m, T+1)
            J_noise = (tmp_noise).dot(time_covar_chol.T)
        else:
            J_noise = np.random.randn(q,m,T+1) 
            
        J = np.zeros([q,m,T+1])
        for i in range(n_ROI_valid):
            for j in range(len(ROI_list[i])):
                J[:,ROI_list[i][j],:] = L_list[i][j]*u[:,i,:] \
                      +J_noise[:, ROI_list[i][j],:]* sigma_J_list[i]
        if n_ROI_valid < n_ROI:
            i = -1
            J[:,ROI_list[i],:] = J_noise[:, ROI_list[i],:]*sigma_J_list[i]
        
        Sigma_E_chol = np.linalg.cholesky(Sigma_E)
        n = G.shape[0]
        noise_M = np.dot(Sigma_E_chol, np.random.randn(q,n,(T+1)))
        M = (np.dot(G, J) + noise_M).transpose([1,0,2])
        u_array = u.transpose([0,2,1])
        y_array = M.transpose([0,2,1])
    
    # soluions
    if False:
        print "using pre-saved simulated data"
        u_array = mat_dict['u'].transpose([0,2,1])
        y_array = mat_dict['M'].transpose([0,2,1])
        
    A_spectral = spectral(y_array, C)
    flag_A_time_vary = True
    p = n_ROI_valid
    Gamma0_0 = np.eye(p)*scale_factor
    A_0 = np.zeros([T,p,p])
    for t in range(T):
        A_0[t] = np.eye(p)*0.9
    Gamma_0 = np.eye(p)*scale_factor
    # more initialization
    sigma_J_list_0 = np.ones(n_ROI)*scale_factor
    L_list_0 = copy.deepcopy(L_list)
    for i in range(n_ROI_valid):
        L_list_0[i] = np.ones(L_list_0[i].size)

    
    #=============try least square  initialization
    R0 = Sigma_E.copy() 
    for l in range(len(sigma_J_list_0)):
        R0 += sigma_J_list_0[l]**2 *  G[:, ROI_list[l]].dot(G[:, ROI_list[l]].T)      
    u_array_hat = get_lsq_u(y_array, R0,C)
    # least sqaure solution  
    Gamma0_ls, A_ls, Gamma_ls = get_param_given_u(u_array_hat, Gamma0_0, A_0, Gamma_0, 
       flag_A_time_vary = flag_A_time_vary,
       prior_Q0 = None, prior_A = None, prior_Q = None,
       MaxIter0 = 100, tol0 = 1E-3, verbose0 = True,
       MaxIter = 100, tol = 1E-6, verbose = True)    
           

    prior_Q0, prior_Q, prior_sigma_J_list = None, None, None
    prior_A = dict(lambda0= 0.0, lambda1 = 1.0)
    #prior_A = None
    MaxIter0, MaxIter = 100,10
    tol0,tol = 1E-5, 1E-3
    verbose0, verbose = False, True
    L_flag = False
    
    obj_true = get_neg_llh_y(y_array, np.linalg.cholesky(Q0),
                             A.copy(),np.linalg.cholesky(Q), 
                            sigma_J_list.copy(), L_list,G, ROI_list, 
                    Sigma_E, prior_Q0 = prior_Q0, 
                   prior_A =prior_A, prior_Q = prior_Q,
                   prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,  
                   L_flag = L_flag, flag_A_time_vary = flag_A_time_vary, eps = 1E-40)
    print "obj_true = %f " % obj_true

    result = EM(
    y_array, G, ROI_list, Sigma_E, 
    Gamma0_0, A_0, Gamma_0, sigma_J_list_0, L_list_0,
    flag_A_time_vary = flag_A_time_vary,
    prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
    prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,
    MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0,
    MaxIter = MaxIter, tol = tol, verbose = verbose, L_flag = L_flag)  

    #Gamma0_0, A_0, Gamma_0 = np.linalg.cholesky(Q0), A.copy(),np.linalg.cholesky(Q)
    #sigma_J_list_0 = sigma_J_list.copy()
    #Q_true = Q.copy()
    #A_true = A.copy()
    #Q0_true = Q0.copy()
    #sigma_J_list_true = sigma_J_list.copy()

    Gamma0_hat, A_hat, Gamma_hat, sigma_J_list_hat, L_list_hat =\
    result['Gamma0'], result['A'], result['Gamma'], result['sigma_J_list'], result['L_list']
    u_t_T_array = result['u_t_T_array']
    obj = get_neg_llh_y(y_array, Gamma0_hat, A_hat, Gamma_hat, sigma_J_list_hat, L_list_hat,
               G, ROI_list, Sigma_E, prior_Q0 = prior_Q0, 
                   prior_A =prior_A, prior_Q = prior_Q,
                   prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,  
                   L_flag = L_flag, flag_A_time_vary = flag_A_time_vary, eps = 1E-40)                 
    obj0 = get_neg_llh_y(y_array, Gamma0_0, A_0, Gamma_0, sigma_J_list_0, L_list,
               G, ROI_list, Sigma_E, prior_Q0 = prior_Q0, 
                   prior_A =prior_A, prior_Q = prior_Q,
                   prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,  
                   L_flag = L_flag, flag_A_time_vary = flag_A_time_vary, eps = 1E-40)
    
    print "obj0 %f, obj %f, obj_true %f\n" %(obj0, obj, obj_true)
    if False:
        result1 = EM(
        y_array, G, ROI_list, Sigma_E, 
        np.linalg.cholesky(Q0), A.copy(),np.linalg.cholesky(Q), 
        sigma_J_list.copy(), L_list_0,
        flag_A_time_vary = flag_A_time_vary,
        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
        prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,
        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0,
        MaxIter = MaxIter, tol = tol, verbose = verbose, L_flag = L_flag)  
      
 
#============== visualize the results:
    for ll in range(1,3):
        if ll == 0:
            i0 = 5
            toplot = [A_hat[i0], A[i0], A[i0]-A_hat[i0]]; names = ['A hat', 'A true', 'A diff'];
        if ll == 1:   
            toplot = [Gamma_hat.dot(Gamma_hat.T), Q, Gamma_hat.dot(Gamma_hat.T)-Q]; names = ['hat', 'true', 'diff'];
        if ll == 2:
            toplot = [Gamma0_hat.dot(Gamma0_hat.T), Q0, Gamma0_hat.dot(Gamma0_hat.T)-Q0]; names = ['hat', 'true', 'diff'];
 
        plt.figure()
        for i in range(3):
            plt.subplot(1,3,i+1); 
            plt.imshow(toplot[i], interpolation = "none", aspect = "auto");
            plt.title(names[i]); plt.colorbar()
    print sigma_J_list, sigma_J_list_hat
    plt.figure()
    trial_id = 2; plt.plot(u_array[trial_id,:,0]); plt.plot(u_t_T_array[trial_id,:,0]);

    corr_ts = 0.0
    # compute the correlation fo the time coureses
    for r in range(q):
        for i in range(p):
            corr_ts += np.corrcoef(u_t_T_array[r,:,i], u_array[r,:,i])[0,1]
    corr_ts /= np.float(q*p)
    print corr_ts
    
    #====== other tests =================
    Gamma0_true1, A_true1, Gamma_true1 = get_param_given_u(u_array, Gamma0_0, A_0, Gamma_0, 
       flag_A_time_vary = flag_A_time_vary,
       prior_Q0 = None, prior_A = None, prior_Q = None,
       MaxIter0 = 100, tol0 = 1E-3, verbose0 = True,
       MaxIter = 100, tol = 1E-6, verbose = True) 
    
    print "error Gamma0 %f" %(np.linalg.norm(Gamma0_hat -Gamma0_true1)/np.linalg.norm(Gamma0_true1))
    print "error A %f" %(np.linalg.norm(A_hat -A_true1)/np.linalg.norm(A_true1))
    print "error Gamma %f" %(np.linalg.norm(Gamma_hat -Gamma_true1)/np.linalg.norm(Gamma_true1))


if True:
    #=============================
    vmin, vmax = -1,1
    m1,m2 = 5,5
    tmpT = 15
    A_list = [A, A_true1, A_ls, A_spectral, A_hat]
    names = ['true','u_array','ls','spectral','hat']
    for l in range(len(A_list)):
        plt.figure()
        for t in range(tmpT):
            _ = plt.subplot(m1,m2,t+1); 
            _ = plt.imshow(A_list[l][t], vmin = vmin, vmax = vmax, interpolation = "none", aspect = "auto")
            _ = plt.colorbar()
        plt.savefig("/home/ying/Dropbox/tmp/%s.png" %names[l])
            
    #================ test if there is some bias in the E step
#    from pykalman import KalmanFilter
#    kf = KalmanFilter( transition_matrices = A_true,
#                      observation_matrices = C,
#                      transition_covariance = Q_true,
#                      observation_covariance = R,
#                      initial_state_covariance = Q0_true,
#                      transition_offsets = np.zeros(p),
#                      observation_offsets = np.zeros(n),
#                        initial_state_mean = np.zeros(p))  
#    Q0_posterior = np.zeros([p,p])
#    for r in range(q):
#        posterior = Kalman_smoothing(T, Q0, A, Q, C, R, y_array[r])
#        Q0_posterior += posterior['P_t_T'][0] \
#            +  np.outer(posterior['u_t_T'][0],posterior['u_t_T'][0])
#        post_mean, post_cov = kf.smooth(y_array[r])
#        # compare with a pre-implemented kalman filter
#        plt.plot(posterior['u_t_T'][:,0]);
#        plt.plot(post_mean[:,0])
#        plt.plot(u_array[r,:,0])
#        diff_cov = posterior['P_t_T']-post_cov
#        plt.plot((np.sum(diff_cov**2, axis = -1)).sum(axis = -1)/np.linalg.norm(post_cov[0]))
#    
#    Q0_posterior /= q
#    toplot = [Q0, Q0_posterior, Q0-Q0_posterior]; 
#    names = ['true','post','diff']
#    plt.figure()
#    for i in range(3):
#        plt.subplot(1,3,i+1); 
#        plt.imshow(toplot[i], interpolation = "none", aspect = "auto");
#        plt.title(names[i]); plt.colorbar()
#    
