# -*- coding: utf-8 -*-

"""
In the simulation and Kalman filter E-step, Q may vary. 
But Q is always fixed in the actual EM model fitting. 
"""
import numpy as np
import copy
#================ simulate Kalman filtered data ==============================
def simulate_kalman_filter_data(T, Q0, A, Q, C, R, n_trial):
    """
    Inputs:
        T+1, total length of the time series
        Q0, initial covariance of the hidden states u
        A, [T,p,p] or [p,p], state transition
        Q, [T,p,p] or [p,p], (time varying) noise covariance for u
        C, [n,p], emission matrix, from hidden states u to observed y
        R, [p,p], noise covariance of y
    Returns:
        u, [n_trial,T+1,p]
        y, [n_trial,T+1,n]
    """
    n,p = C.shape
    flag_A_time_vary = True if len(A.shape)>2 else False
    flag_Q_time_vary = True if len(Q.shape)>2 else False
    u = np.zeros([n_trial, T+1,p])
    y = np.zeros([n_trial, T+1,n])
    for r in range(n_trial):
        u[r,0] = np.random.multivariate_normal(np.zeros(p), Q0)
        y[r,0] = np.random.multivariate_normal(np.zeros(n), R) + C.dot(u[r,0])
        for t in range(1,T+1):
             tmpA = A[t-1].copy() if flag_A_time_vary else A.copy()
             tmpQ = Q[t-1].copy() if flag_Q_time_vary else Q.copy()
             u[r,t] = np.random.multivariate_normal(np.zeros(p), tmpQ) + tmpA.dot(u[r,t-1])
             y[r,t] = np.random.multivariate_normal(np.zeros(n), R) + C.dot(u[r,t])        
    return u, y            
#================ Kalman smoothing for single trials ==========================
def Kalman_smoothing(T, Q0, A, Q, C, R, y):
    """
    Inputs:
        T+1, total length of the time series
        Q0, initial covariance of the hidden states u
        A, [T,p,p] or [p,p], state transition
        Q, [T,p,p] or [p,p], (time varying) noise covariance for u
        C, [n,p], emission matrix, from hidden states u to observed y
        R, [p,p], noise covariance of y
        y, [T+1,n], single trial observed 
    Returns:
        dict
        u_t_T, [T+1,p] posterior mean 
        p_t_T, [T+1,p,p] posterior cov of u_t
        p_tt_T, [T+1,p,p] posterior cross cov of u_t, u_{t-1}
    """
    
    n,p = C.shape
    flag_A_time_vary = True if len(A.shape)>2 else False
    flag_Q_time_vary = True if len(Q.shape)>2 else False
    #== declare variables  ==
    u_t_t_1 = np.zeros([T+1,p])
    u_t_t = np.zeros([T+1,p]) 
    P_t_t_1 = np.zeros([T+1,p,p])
    P_t_t = np.zeros([T+1,p,p])
    K_t = np.zeros([T+1,p,n])
    H_t = np.zeros([T+1,p,p])
    u_t_T = np.zeros([T+1,p])
    P_t_T = np.zeros([T+1,p,p])
    # P_{t,t-1}^T #  length is (T+1), but only 1:T is relevent
    P_tt_T = np.zeros([T+1,p,p])
    
    #== foward operations ==
    # initialization
    # these two should not be used
    u_t_t_1[0] = np.zeros(p) 
    P_t_t_1[0]= Q0.copy()
    # Stoffer&Shumway did not include the 0 time point, but in their case, y0 is not observed
    # according to definition, time 0 should also be updated
    tmp_inv = np.linalg.inv(reduce(np.dot, [C,Q0,C.T]) + R)
    u_t_t[0] = reduce(np.dot, [Q0, C.T, tmp_inv, y[0]] )
    P_t_t[0] = Q0- reduce(np.dot, [Q0,C.T,tmp_inv, C,Q0])
    
    # forward iterations
    # forward prediction
    for t in range(1,T+1):
        # obtain the current A and Q
        # A and Q have length T, their index should be t-1
        tmpA = A[t-1].copy() if flag_A_time_vary else A.copy()
        tmpQ = Q[t-1].copy() if flag_Q_time_vary else Q.copy()
        # compute the posterior means and covs
        u_t_t_1[t,:] = tmpA.dot(u_t_t[t-1,:])
        P_t_t_1[t] = reduce(np.dot, [tmpA, P_t_t[t-1], tmpA.T]) + tmpQ
        tmp_inv = np.linalg.inv(reduce(np.dot, [C,P_t_t_1[t],C.T]) + R)
        K_t[t] = reduce(np.dot, [P_t_t_1[t], C.T, tmp_inv])
        u_t_t[t] = u_t_t_1[t] + K_t[t].dot( y[t] - C.dot(u_t_t_1[t]) )
        P_t_t[t] = P_t_t_1[t] - reduce(np.dot, [ K_t[t], C, P_t_t_1[t] ])
        # compute H_t for backward use
        H_t[t-1] = reduce(np.dot, [P_t_t[t-1], tmpA.T, np.linalg.inv(P_t_t_1[t])])
    
        #==backward operations==
    # initialization
    u_t_T[T] = u_t_t[T].copy()
    P_t_T[T] = P_t_t[T].copy()
    tmpA = A[T-1].copy() if flag_A_time_vary else A.copy()
    tmpQ = Q[T-1].copy() if flag_Q_time_vary else Q.copy()
    P_tt_T[T] = (np.eye(p)- K_t[T].dot(C)).dot(tmpA.dot(P_t_t[T-1]))
    # backward iterations
    for t in range(T,0,-1):
        # obtain the current A and Q
        # A and Q have length T, their index should be t-1
        tmpA = A[t-1].copy() if flag_A_time_vary else A.copy()
        tmpQ = Q[t-1].copy() if flag_Q_time_vary else Q.copy()
        # here
        u_t_T[t-1] = u_t_t[t-1] + H_t[t-1].dot(u_t_T[t]-tmpA.dot(u_t_t[t-1]))
        P_t_T[t-1] = P_t_t[t-1] + reduce(np.dot,\
                                 [ H_t[t-1], (P_t_T[t]-P_t_t_1[t]), H_t[t-1].T])
        if t>= 2:                                 
            P_tt_T[t-1] = P_t_t[t-1].dot(H_t[t-2].T) + reduce(np.dot,\
                     [H_t[t-1], (P_tt_T[t]- tmpA.dot(P_t_t[t-1])), H_t[t-2].T])     
    posterior = dict(u_t_T = u_t_T, P_t_T = P_t_T, P_tt_T = P_tt_T)
    return posterior
 
#===========================
# optimization for the M step
# (1) optimization of Q0
# (2) coordinate descent of sigma_J_list, L_list
# (3) coordinate descent of Q_t, A_t, also with the option of fixed Q, A
 
#======= utility of matrix covarience and det
def _util_cov_inv_det(S, eps = 1E-40):
    # inv_S, log_det = _util_cov_inv_det(S, eps)
    tmp_u,tmp_d,tmp_v = np.linalg.svd(S)
    tmp_u = (tmp_u+tmp_v.T)/2.0
    tmp_d[tmp_d <= 0] = eps           
    log_det = np.sum(np.log(tmp_d))
    inv_S = (tmp_u/tmp_d).dot(tmp_u.T)
    eigs = tmp_d.copy()
    return inv_S, log_det, eigs
  
 
#===== compute the posterior matrices B1,B2,B3,B4,B5,B6, and
# B7 = sum_q P_0_T of q trials, which are not dependent on the parameters ====
def get_posterior_matrix(u_t_T_array, P_t_T_array, P_tt_T_array, y_array):
    """
    Inputs:
        u_t_T_array: [q,T+1,p]  u_t_T [T+1,p] of q trials
        P_t_T_array: [q,T+1,p,p] P_t_T [T+1,p,p] of q trials
        P_tt_T_array: [q,T+1,p,p] P_tt_T [T+1,p,p] of q_trials
        y_array, [q,T+1,n],  y [T+1,n], sensor time series data of q trials
    Outputs:
        dict, B_mat, 
        B1_B3, [T+1,p,p] \sum_q (P_t_T[t]  + u_t_T[t], u_t_T[t]^T), B1 1:T+1, B3 = 0:T
        B2, [T+1,p,p] \sum_q (P_tt_T[t+1] + u_t_T[t+1], u_t_T[t]^T) # B2[T] is 0
        B4, [n,n] \sum_q \sum_t y[t] y[t]^T
        B5, [n,p] \sum_q \sum_t y[t] u_t_T[t]^T
        B6, [p,p] \sum_q \sum_t (u_t_T[t] u_t_T[t]^T + P_t_T[t]) = sum of B1_B3 for all T
        B7, [p,p] \sum_q, P_0_T + u_t_T[0] u_t_T[t]^T
        q = q,  keep q in for convenience.
    """
    q,T0,n = y_array.shape
    p = u_t_T_array.shape[2]
    T = T0-1
    B4 = np.zeros([n,n]) # B4 = sum_t r y_t y_t^T
    B5 = np.zeros([n,p]) # B5 = sum_t_r y_r u_t^T  #  B5 C^T
    B1_B3 = P_t_T_array.sum(axis = 0) # \sum_q (P_t_T[t]
    B2 = P_tt_T_array.sum(axis = 0)
    for t in range(T+1):
        # \sum_q u_t_T[t], u_t_T[t]^T
        B1_B3[t] += u_t_T_array[:,t,:].T.dot(u_t_T_array[:,t,:])
        # \sum_q u_t_T[t+1], u_t_T[t]^T
        if t+1 <= T:
            B2[t] += u_t_T_array[:,t+1,:].T.dot(u_t_T_array[:,t,:])
        B4 += y_array[:,t,:].T.dot(y_array[:,t,:])
        B5 += y_array[:,t,:].T.dot(u_t_T_array[:,t,:])
    # B6 = \sum_t \sum_q u_t u_t^t + P_t_T
    B6 = B1_B3.sum(axis = 0)
    B7 = P_t_T_array[:,0,:,:].sum(0) + u_t_T_array[:,0,:].T.dot(u_t_T_array[:,0,:])# B7 = \sum_q P_0_T
    B_mat = dict(B1_B3= B1_B3, B2 = B2, B4 = B4, B5 = B5, B6 = B6, B7 = B7, q = q)
    return B_mat            

#=============== priors for Q and sigma_j
def obj_prior_Q(Gamma, prior_Q, eps = 1E-40):
    """
    prior_Q: dict,  'nu' >p, 
                    'V', [p,p],  V for inverse Wishart, V_inv for Wishart
                    'flag_inv', if true, inverse Wishart, else, V is V^{-1} for Wishart              
    """
    tmp_Q = Gamma.dot(Gamma.T)
    inv_Q, log_det, eigs = _util_cov_inv_det(tmp_Q, eps = eps)
    #u, eigs0, _ = np.linalg.svd(tmp_Q)
    #eigs = eigs0.copy()
    #eigs[eigs <= 0] += eps
    #log_det = np.sum(np.log(eigs))
    nu = np.float(prior_Q['nu'])
    V = prior_Q['V']
    p = np.float(Gamma.shape[0])
    if prior_Q['flag_inv']:
        #inv_Q = (u /eigs).dot(u.T) 
        obj = (nu+p+1)*log_det + np.sum(V*inv_Q)
    else:
        inv_V = V
        obj = -(nu-p-1)*log_det + np.sum(inv_V*tmp_Q)
    return obj
    
def grad_prior_Q(Gamma, prior_Q, eps = 1E-40):
    
    tmp_Q = Gamma.dot(Gamma.T)
    inv_Q, log_det, eigs = _util_cov_inv_det(tmp_Q, eps = eps)
    #u, eigs0, _ = np.linalg.svd(tmp_Q)
    #eigs = eigs0.copy()
    #inv_Q = (u /eigs).dot(u.T)
    nu = np.float(prior_Q['nu'])
    V = prior_Q['V']
    p = np.float(Gamma.shape[0])
    if prior_Q['flag_inv']:
        grad = 2.0*(inv_Q.dot(np.eye(p)*(nu+p+1)-V.dot(inv_Q))).dot(Gamma)
    else:
        inv_V = V
        grad = 2.0*(-(nu-p-1)*inv_Q + inv_V ).dot(Gamma)
    return grad

def obj_prior_sigma_J_list(sigma_J_list, prior_sigma_J_list):
    """
    prior_sigma_J_list, dict, 'alpha'>0, 'beta'>0, 'flag_inv', if true use inverse gamma
    """
    alpha, beta= np.float(prior_sigma_J_list['alpha']), np.float(prior_sigma_J_list['beta'])
    if not prior_sigma_J_list['flag_inv']: #gamma
        obj = -2.0*np.sum((alpha-1.0)* np.log(sigma_J_list**2) - beta* sigma_J_list**2)
    else:
        obj = 2.0*np.sum((alpha+1.0)* np.log(sigma_J_list**2) + beta/(sigma_J_list**2))
    return obj
    
def grad_prior_sigma_J_list(sigma_J_list, prior_sigma_J_list):
    """
    prior_sigma_J_list, dict, 'alpha'>0, 'beta'>0, 'flag_inv', if true use inverse gamma
    """
    alpha, beta= np.float(prior_sigma_J_list['alpha']), np.float(prior_sigma_J_list['beta'])
    if not prior_sigma_J_list['flag_inv']: # gamma
        grad = -4.0*((alpha-1.0)/sigma_J_list- beta*sigma_J_list)
    else:
        grad = 4.0*((alpha+1.0)/sigma_J_list - beta/(sigma_J_list**3))
    return grad


#============= optimization of RC parts =======================================       
#======== utility function shared by obj and grad for RC
def _util_obj_grad_RC(sigma_J_list, L_list, B4, B5, B6, ROI_list, G, Sigma_E, eps):

    GL = np.zeros([G.shape[0],len(L_list)])
    for l in range(len(L_list)):
        GL[:,l] = G[:,ROI_list[l]].dot(L_list[l])    
    tmp = B5.dot(GL.T)
    B0 = B4- tmp - tmp.T + reduce(np.dot, [GL, B6, GL.T])           
    R = Sigma_E.copy() 
    for l in range(len(sigma_J_list)):
        R += sigma_J_list[l]**2 *  G[:, ROI_list[l]].dot(G[:, ROI_list[l]].T)     
    R_inv, log_det, eigs = _util_cov_inv_det(R, eps = eps)
    return GL, B0, R, eigs, R_inv       
     
#======== objective function for RC, i.e. sigma_J_list and L_list ============
def obj_RC( sigma_J_list, L_list,  B4, B5, B6, ROI_list, G, Sigma_E, T, q,
            prior_sigma_J_list, prior_L_precision, L_flag = True, eps = 1E-40):
    """
        Objective function of the RC   minimize (T + 1)q log det(R) + trace(R^−1 B0 )}  
                    B0 = B4 −B5 L^T G^T −GL B5^T +GL B6 L^T G^T
        Inputs: 
            sigma_J_list  [p+1,1]
            L_list  [p], list of arrays
            prior_L_precision, [p], list of precision of the spatial smoothing cov
            prior_sigma_J_list, dictionary, alpha, beta, for sigma_j^2            
    """
    n,m = G.shape
    GL, B0, R, eigs, R_inv = _util_obj_grad_RC(sigma_J_list, L_list,  
                                               B4, B5, B6, ROI_list, G, Sigma_E, eps) 
    log_det_R = np.sum(np.log(eigs))
    trace_R_inv_B0 = np.sum(R_inv * B0)
    nllh = (T+1)*q*log_det_R + trace_R_inv_B0
    if prior_sigma_J_list is not None:
        nllh += obj_prior_sigma_J_list(sigma_J_list, prior_sigma_J_list)
    if prior_L_precision is not None and L_flag:
        # 1/2 in the llh was all dropped
        for i in range(len(L_list)):
            nllh += np.dot(prior_L_precision[i], L_list[i]).T.dot(L_list[i])
    return nllh

#========= gradient function for RC, i.e., sigma_J_list and L_list
def grad_RC( sigma_J_list, L_list,  B4, B5, B6, ROI_list, G, Sigma_E, T, q,
            prior_sigma_J_list, prior_L_precision,
            sigma_J_flag = True, L_flag = True, eps = 1E-40):
            
    n,m = G.shape
    p = B6.shape[0]
    grad_sigma_J_list = np.zeros(sigma_J_list.shape)
    grad_L_list = list()
    for i in range(p):
        grad_L_list.append(np.zeros(L_list[i].size))    
    GL, B0, R, eigs, R_inv = _util_obj_grad_RC(sigma_J_list, L_list,  
                                               B4, B5, B6, ROI_list, G, Sigma_E, eps = eps)  
    if sigma_J_flag:
        # R^{-1} q(T+1)- R^{-1} B0 R^{-1}
        grad_R = R_inv.dot( np.float(T+1)*np.float(q)*np.eye(n)- B0.dot(R_inv))
        for i in range(len(sigma_J_list)):
            tmpGGT = np.sum( grad_R*(np.dot(G[:,ROI_list[i]], G[:,ROI_list[i]].T)))
            grad_sigma_J_list[i] = 2* sigma_J_list[i]*tmpGGT  
            if prior_sigma_J_list is not None:
                grad_sigma_J_list += grad_prior_sigma_J_list(sigma_J_list, prior_sigma_J_list)
    if L_flag:
        #2G^T R^{−1} (−B5 + GL B6 )
        tmp = R_inv.dot(-B5 + GL.dot(B6))
        grad_L0 = 2.0* G.T.dot(tmp)
        for i in range(p):
            grad_L_list[i] = grad_L0[ROI_list[i],i]
            if prior_L_precision is not None:
                grad_L_list[i] += 2.0* prior_L_precision[i].dot(L_list[i])  
    return grad_sigma_J_list, grad_L_list
 
# ==== accelarated gradient descent with back tracking, RC=========
def grad_descent_RC(sigma_J_list_0, L_list_0,  B4, B5, B6, ROI_list, G, Sigma_E, T, q,
                    prior_sigma_J_list, prior_L_precision,
                    sigma_J_flag = True, L_flag = True,
                      tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6,
                      eps = 1E-40, verbose = True):
                          
    diff_obj, obj, old_obj = np.inf, 1E10, 0
    IterCount = 0
    sigma_J_list = sigma_J_list_0.copy()
    L_list =copy.deepcopy(L_list_0)
    
    p = len(L_list)
    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break                  
        grad_sigma_J_list, grad_L_list =  grad_RC(sigma_J_list, L_list,  
                            B4, B5, B6, ROI_list, G, Sigma_E, T, q,
                            prior_sigma_J_list, prior_L_precision,
                            sigma_J_flag = sigma_J_flag, L_flag = L_flag, eps = eps)                    
        f = obj_RC( sigma_J_list, L_list,  B4, B5, B6, ROI_list, G, Sigma_E, T, q,
                       prior_sigma_J_list, prior_L_precision, L_flag = L_flag, eps = eps)
        step = step_ini
        tmp_diff = np.inf
        while tmp_diff > 0:
            step *= tau
            ref = f -step/2.0*(np.sum(grad_sigma_J_list**2))
            for i in range(p):
                ref = ref-step/2.0*(np.sum(grad_L_list[i]**2))

            tmp_sigma_J_list = sigma_J_list-step*grad_sigma_J_list*np.float(sigma_J_flag) 
            tmp_L_list = copy.deepcopy(L_list)
            for i in range(p):
                tmp_L_list[i] -= step*grad_L_list[i]*np.float(L_flag) 
            
            tmp_f = obj_RC( tmp_sigma_J_list, tmp_L_list, B4, B5, B6,
                           ROI_list, G, Sigma_E, T, q,
                       prior_sigma_J_list, prior_L_precision, L_flag = L_flag, eps = eps)            
            tmp_diff = tmp_f-ref
        
        sigma_J_list = tmp_sigma_J_list.copy()
        L_list = copy.deepcopy(tmp_L_list)
        old_obj, obj = obj, tmp_f
        diff_obj = old_obj-obj
        if verbose:
            print "Iter %d \n, obj=%f\n diff_obj = %f\n" %(IterCount,obj,diff_obj)
        IterCount += 1
    return sigma_J_list, L_list, obj

#=======coor descent of RC ==========================================               
def coor_descent_RC(sigma_J_list_0, L_list_0,  B4, B5, B6, ROI_list, G, Sigma_E, T, q,
                    prior_sigma_J_list, prior_L_precision,
                    sigma_J_flag = True, L_flag = True,
                      tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6,
                      eps = 1E-40, verbose = True, 
                      MaxIter0 = 100, tol0 = 1E-3, verbose0 = False):
    # initialization                     
    diff_obj, obj, old_obj = np.inf, 1E10, 0
    IterCount = 0
    sigma_J_list = sigma_J_list_0.copy()
    L_list =copy.deepcopy(L_list_0)
    # iterations
    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break
        old_obj = obj
        # update each variable with the wrapped function
        # flag_list, current [ sigma_J_flag, L_flag]
        if sigma_J_flag and L_flag:
            flag_list = [[True, False], [False, True]]
        elif sigma_J_flag:
            flag_list = [[True, False]]
        elif L_flag:
            flag_list = [[False, True]]
        else:
            raise ValueError("At least one of the flags must be true")
            
        for l in range(len(flag_list)):
            tmp_sigma_J_flag, tmp_L_flag = flag_list[l][0], flag_list[l][1]
            if verbose:
                print "sigma_J_flag %d L_flag %d" %(tmp_sigma_J_flag, tmp_L_flag)
            sigma_J_list, L_list, obj =grad_descent_RC( sigma_J_list, L_list,  
                    B4, B5, B6, ROI_list, G, Sigma_E, T, q,
                    prior_sigma_J_list, prior_L_precision,
                    sigma_J_flag = tmp_sigma_J_flag, L_flag = tmp_L_flag,
                      tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                      eps = eps, verbose = verbose0) 
        diff_obj = old_obj-obj
        if verbose:
            print "coordinate descent Iter %d \n obj = %f\n diff_obj = %f\n" %(IterCount,obj,diff_obj)
        IterCount += 1
    return sigma_J_list, L_list, obj

#========== utility function for AQ, do svd and get the inverse
def _util_obj_grad_AQ(tmp_Gamma):
    tmp_u, tmp_d, tmp_v = np.linalg.svd(tmp_Gamma)
    eig_Q_sqrt = tmp_d 
    inv_Q = np.dot(tmp_u/tmp_d**2, tmp_u.T)
    return eig_Q_sqrt, inv_Q
    
def _util_obj_grad_AQ_sum_B123(A, B1_B3, B2, T, flag_A_time_vary):
    tmp = B1_B3[1:(T+1)].sum(axis = 0)
    if flag_A_time_vary:
        for t in range(T):
            tmp_BA = A[t].dot(B2[t].T)
            tmp += - tmp_BA - tmp_BA.T + reduce(np.dot, [A[t], B1_B3[t], A[t].T ])
    else:
        tmp_BA = A.dot((B2.sum(axis = 0)).T)
        tmp += -tmp_BA - tmp_BA.T + reduce(np.dot, [A, B1_B3[0:T].sum(axis = 0), A.T ])
    return tmp
 
#===========objective function for AQ ============
def obj_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q, 
           flag_A_time_vary):
    #\sum_T_[q log det(Q t ) + trace(Q_t^{-1} (B 1t − A t B_2^T − B 2t A t + A t B 3t A t ))
    # Gamma (Q) is fixed.   [p,p]  
    """
    prior_A: dict, ['lambda0','lambda1'] 
            lambda0||A_t||_F^2 + lambda1||A_t-A_t_1||_F^2 
            If A is fixed, lambda1 is automatically 0   
    """
    eig_Q_sqrt, inv_Q = _util_obj_grad_AQ(Gamma)
    log_det_Q = 2*np.sum(np.log(eig_Q_sqrt))
    tmp = _util_obj_grad_AQ_sum_B123(A, B1_B3, B2, T, flag_A_time_vary)
    nllh = np.float(q)*np.float(T)*log_det_Q + np.sum(inv_Q*tmp)
    if prior_A is not None:
        nllh += prior_A['lambda0']* np.sum(A**2)
        if flag_A_time_vary:
            for t in range(T-1):
                nllh += prior_A['lambda1']*np.sum((A[t+1]-A[t])**2)     
    # if prior_Q is not None, add the prior       
    if prior_Q is not None:
        nllh += obj_prior_Q(Gamma, prior_Q)
    return nllh 
#=========== gradient funciton for AQ ============
def grad_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary,
            A_flag = True, Q_flag = True):
    grad_A = np.zeros(A.shape)
    grad_Gamma = np.zeros(Gamma.shape)
    eig_Q_sqrt, inv_Q = _util_obj_grad_AQ(Gamma)
    if A_flag:
        if flag_A_time_vary:
            for t in range(T):
                # 2 Q^-1(-B2 + A B_3)
                grad_A[t] = 2.0*np.dot(inv_Q, (-B2[t]+ A[t].dot(B1_B3[t])))
                if prior_A is not None:
                    # gradient of lambda0||A_t||_F^2 + lambda1||A_t-A_t_1||_F^2 
                    # 2 lambda0 A_t + 2 lambda1 ((A_t_1 + A_t+1)-2 A_t)                    
                    grad_A[t] += 2.0* prior_A['lambda0'] *A[t]
                    if t > 0 and t < T-1:
                        #grad_A[t] += 2.0*prior_A['lambda1']*(A[t-1]+A[t+1]-2.0*A[t])
                        # correction on May 19
                        grad_A[t] += 2.0*prior_A['lambda1']*(2.0*A[t]-A[t-1]-A[t+1])
                    elif t == 0:
                        grad_A[t] += 2.0*prior_A['lambda1']*(A[0]-A[1])
                    else: # t = t-1
                        grad_A[t] += 2.0*prior_A['lambda1']*(A[t]-A[t-1])
        else:
            grad_A = 2.0*np.dot(inv_Q, (-B2.sum(axis=0)+A.dot((B1_B3[0:T]).sum(axis = 0))))
            if prior_A is not None:
                grad_A += 2.0*prior_A['lambda0']        
    if Q_flag:
        # gradient = Q^{-1}(qT I-\sum_T B1t-AtB2t^T-B2tAt^T + AtB3tAt^T)Q^{-1}) Gamma
        tmp = _util_obj_grad_AQ_sum_B123(A, B1_B3, B2, T, flag_A_time_vary)
        grad_Gamma = reduce(np.dot, [inv_Q, (np.eye(inv_Q.shape[0])*np.float(q)*np.float(T)-tmp.dot(inv_Q)), Gamma])
        if prior_Q is not None:
            grad_Gamma += grad_prior_Q(Gamma, prior_Q)
        grad_Gamma = np.tril(grad_Gamma)                    
    return grad_A, grad_Gamma

#=========== gradient descent with back tracking for AQ =============
def grad_descent_AQ(A_0, Gamma_0, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary,
                    A_flag = True, Q_flag = True,
                      tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6,
                      verbose = True):
                          
    diff_obj, obj, old_obj = np.inf, 1E10, 0
    IterCount = 0
    A, Gamma = A_0.copy(), Gamma_0.copy()
    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break                  
        grad_A, grad_Gamma = grad_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary,
                                     A_flag = A_flag, Q_flag = Q_flag)                    
        f = obj_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary)
        step = step_ini
        tmp_diff = np.inf
        while tmp_diff > 0:
            step *= tau
            ref = f -step/2.0*(np.sum(grad_A**2)+np.sum(grad_Gamma**2))
            tmp_A = A - step*grad_A*np.float(A_flag) 
            tmp_Gamma = Gamma- step*grad_Gamma*np.float(Q_flag) 
            tmp_f = obj_AQ(tmp_A, tmp_Gamma, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary)           
            tmp_diff = tmp_f-ref
        A, Gamma = tmp_A.copy(), tmp_Gamma.copy()
        old_obj, obj = obj, tmp_f
        diff_obj = old_obj-obj
        if verbose:
            print "Iter %d \n, obj=%f\n diff_obj = %f\n" %(IterCount,obj,diff_obj)
        IterCount += 1
    return A, Gamma, obj
    
#=========== coordinate descent for AQ ===========               
def coor_descent_AQ(A_0, Gamma_0, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary,
                    A_flag = True, Q_flag = True,
                      tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6,
                      verbose = True,
                      MaxIter0 = 100, tol0 = 1E-3, verbose0 = False):
    """
    if prior_Q is None, then each with each update, 
    Q_hat = tmp (A)/qT, 
    """                      
    # initialization                     
    diff_obj, obj, old_obj = np.inf, 1E10, 0
    IterCount = 0
    A, Gamma = A_0.copy(), Gamma_0.copy()
    if (not A_flag) and (not Q_flag):
        raise ValueError("At least one of the flags must be true")
    # iterations
    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break
        old_obj = obj
        # update each variable with the wrapped function
        # flag_list, current [ A_flag, Q_flag]
        if prior_Q is not None:
            if A_flag and Q_flag:
                flag_list = [[False, True],[True, False]]
            elif A_flag:
                flag_list = [[True, False]]
            else:
                flag_list = [[False, True]]
            for l in range(len(flag_list)):
                tmp_A_flag, tmp_Q_flag = flag_list[l][0], flag_list[l][1]
                if verbose:
                    print "A_flag %d Q_flag %d" %(tmp_A_flag, tmp_Q_flag)
                A, Gamma, obj = grad_descent_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary,
                            A_flag = tmp_A_flag, Q_flag = tmp_Q_flag,
                          tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                          verbose = verbose0)
        else: # prior_Q is None:
            if Q_flag:
                tmp = _util_obj_grad_AQ_sum_B123(A, B1_B3, B2, T, flag_A_time_vary)
                tmpQ = tmp/np.float(q)/np.float(T)
                Gamma = np.linalg.cholesky(tmpQ)
                obj = obj_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q,flag_A_time_vary)
            if A_flag:
                A, Gamma, obj = grad_descent_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary,
                            A_flag = True, Q_flag = False,
                          tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                          verbose = verbose0)
        diff_obj = old_obj-obj
        if verbose:
            print "coordinate descent Iter %d \n obj = %f\n diff_obj = %f\n" %(IterCount,obj,diff_obj)
        IterCount += 1
    return A, Gamma, obj       
    
#=========== obtimization for Q0==============
# note Gamma0 is the cholesky decomposition of Q0, Gamma_0 is the cholesky decompsition of Gamma
def obj_Q0(Gamma0, B7, q, prior_Q0):
    # q log det(Q 0 ) + trace(Q_0^−1 P_0|T)
    eig_Q_sqrt, inv_Q = _util_obj_grad_AQ(Gamma0)
    log_det_Q0 = 2*np.sum(np.log(eig_Q_sqrt))
    nllh = np.float(q)*log_det_Q0+ np.sum(inv_Q*B7)
    if prior_Q0 is not None:
        nllh += obj_prior_Q(Gamma0, prior_Q0)
    return nllh
#  gradient 
def grad_Q0(Gamma0, B7, q, prior_Q0):
    eig_Q_sqrt, inv_Q = _util_obj_grad_AQ(Gamma0)
    grad_Gamma0 = reduce(np.dot, [inv_Q, np.float(q)*np.eye(Gamma0.shape[0])- B7.dot(inv_Q), Gamma0])
    if prior_Q0 is not None:
        grad_Gamma0 += grad_prior_Q(Gamma0, prior_Q0)
    grad_Gamma0 = np.tril(grad_Gamma0)
    return grad_Gamma0
# gradient descent    
def grad_decent_Q0(Gamma0_0, B7, q, prior_Q0, tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6,
                      verbose = True):
    diff_obj, obj, old_obj = np.inf, 1E10, 0
    IterCount = 0
    Gamma0 = Gamma0_0.copy()
    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break                  
        grad_Gamma0 = grad_Q0(Gamma0,B7,q,prior_Q0)                     
        f = obj_Q0(Gamma0, B7, q, prior_Q0)
        step = step_ini
        tmp_diff = np.inf
        while tmp_diff > 0:
            step *= tau
            ref = f -step/2.0*(np.sum(grad_Gamma0**2))
            tmp_Gamma0 = Gamma0-step*grad_Gamma0
            tmp_f = obj_Q0(tmp_Gamma0, B7, q, prior_Q0)          
            tmp_diff = tmp_f-ref
        Gamma0 = tmp_Gamma0.copy()
        old_obj, obj = obj, tmp_f
        diff_obj = old_obj-obj
        if verbose:
            print "Iter %d \n, obj=%f\n diff_obj = %f\n" %(IterCount,obj,diff_obj)
        IterCount += 1
    return Gamma0, obj
#========= The M step=====
def update_param_M_step(B_mat, G, ROI_list, Sigma_E,
                        Gamma0_0, A_0, Gamma_0, sigma_J_list_0, L_list_0, 
                        flag_A_time_vary = False,
                        prior_Q0 = None, prior_A = None, prior_Q = None,
                        prior_L_precision = None, prior_sigma_J_list = None,
                        MaxIter0 = 100, tol0 = 1E-3, verbose0 = False,
                        MaxIter = 100, tol = 1E-6, verbose = True, L_flag = False):
    """
    Update the parameters in the M step
    Inputs:
        B_mat, the posterior matrix dictionary, B1_B3, B2, B4, B5, B6, B7
        G, [n,m] forward matrix, m dipoles and n sensors
        ROI_list, [p+1,] list of ROI indices corresponding to columns of G  
        Sigma_E, [n,n], sensor noise cov
        flag_time_vary, Boolean, if False, A and Q are fixed, A, [p,p], Q,[p,p]
        #== priors: if None, no priors needed
            prior_Q0,  dictionary of inverse gamma/gamma priors [V,nu] (TBA)
            prior_A, shrinking and smoothing priors,  
                    \lambda1 ||A[t]-A[t-1]||_F^2 + \lambda0 ||A[t]||_F^2
            prior_Q, smoothing priors + inverse gamma/ gamma
                    \mu1 ||Gamma[t]-Gamma[t-1]||_F^2 + TBA
            prior_L_precision, [p], list of precision of the spatial smoothing cov
            prior_sigma_J_list, dictionary, alpha, beta, for sigma_j^2
             
    Outputs:
        sigma_J_list, [p+1],    R = Sigma_E + G sigma_J_list.^2 G^T
        L_list, [p], list of L arrays,  C = GL
        Gamma0, [p,p], cholosky decomposition of intial cov Q
        A, [T,p,p] or [p,p], state transition
        Gamma, [T,p,p] or [p,p], cholesky decompositions of (time varying) noise covariance Q
    """
    q, B1_B3, B2, B4, B5, B6, B7 = B_mat['q'], B_mat['B1_B3'],  B_mat['B2'],\
                           B_mat['B4'], B_mat['B5'], B_mat['B6'], B_mat['B7']
    T = B1_B3.shape[0]-1  
    tau, step_ini, eps = 0.8,1.0, 1E-40                   
    #== update Q0
    # if no prior, directly update it :
    if prior_Q0 is None:
        Q0 = B7/np.float(q)
        Gamma0 = np.linalg.cholesky(Q0)
        obj_for_Q0 = obj_Q0(Gamma0, B7, q, prior_Q0)
    else:
        Gamma0, obj_for_Q0 = grad_decent_Q0(Gamma0_0, B7, q, prior_Q0, tau = tau,
                      step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                      verbose = verbose0)                
    #== update At, Qt
    if prior_A is None and prior_Q is None:
        if flag_A_time_vary:
            p = len(L_list_0)
            A = np.zeros([T,p,p])
            for t in range(T):
                A[t] = B2[t].dot( np.linalg.inv(B1_B3[t])) 
        else:
            A = B2.sum(axis = 0).dot( np.linalg.inv( (B1_B3[0:T].sum(axis = 0)) ))
        # Q = 1/q (\sum_B1 - A \sum B2^T - \sum B2 A^T + A \sum B3 A^T)
        tmp = _util_obj_grad_AQ_sum_B123(A, B1_B3, B2, T, flag_A_time_vary)
        Q = tmp/np.float(q)/np.float(T)
        Gamma = np.linalg.cholesky(Q)
        obj_for_AQ = obj_AQ(A, Gamma, B1_B3, B2, T,q, prior_A, prior_Q, flag_A_time_vary)
    else:  # coordinate descent
        A, Gamma, obj_for_AQ = coor_descent_AQ(A_0, Gamma_0, B1_B3, B2, T,q, prior_A, prior_Q, 
                  flag_A_time_vary, A_flag = True, Q_flag = True, tau = tau, 
                  step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                  verbose = verbose0)  
    #== update sigma_J_list, L_list, coordinate descent 
    sigma_J_list, L_list, obj_for_RC = coor_descent_RC(sigma_J_list_0, L_list_0, 
                    B4, B5, B6, ROI_list, G, Sigma_E, T, q,
                    prior_sigma_J_list, prior_L_precision,
                    sigma_J_flag = True, L_flag = L_flag,
                      tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                      eps = eps, verbose = verbose0) 
                      
    obj_Mstep = obj_for_RC + obj_for_AQ + obj_for_Q0 
    result = dict(Gamma0 = Gamma0, A = A, Gamma = Gamma, 
                  sigma_J_list = sigma_J_list, L_list = L_list,
                  obj_Mstep = obj_Mstep)                  
    return result
    
#============================================================================= 
def EM(y_array, G, ROI_list, Sigma_E, Gamma0_0, A_0, Gamma_0, sigma_J_list_0, L_list_0,
       flag_A_time_vary = False,
       prior_Q0 = None, prior_A = None, prior_Q = None,
       prior_L_precision = None, prior_sigma_J_list = None,
       MaxIter0 = 100, tol0 = 1E-3, verbose0 = True,
       MaxIter = 100, tol = 1E-6, verbose = True, L_flag = False):
    q,T,n = y_array.shape  
    T = T-1
    Gamma0 = Gamma0_0.copy()
    A = A_0.copy()
    Gamma = Gamma_0.copy()
    sigma_J_list = sigma_J_list_0.copy()
    L_list = copy.deepcopy(L_list_0)
    
    p = len(L_list)
    diff_param = np.inf
    diff_obj = np.inf
    obj = 1E8
    IterCount = 0
    # iteration TBA
    while diff_param > tol or np.abs(diff_obj) > tol:
        if IterCount > MaxIter:
            print " EM MaxIter achieved"
            break
        # E step
        GL = np.zeros([n,p])
        for l in range(len(L_list)):
            GL[:,l] = G[:,ROI_list[l]].dot(L_list[l])              
        R = Sigma_E.copy() 
        for l in range(len(sigma_J_list)):
            # debug
            #print ROI_list[l]
            #print sigma_J_list[l]
            #print G[:, ROI_list[l]].shape
            R += sigma_J_list[l]**2 *  G[:, ROI_list[l]].dot(G[:, ROI_list[l]].T) 
        Q = Gamma.dot(Gamma.T) 
        u_t_T_array = np.zeros([q,T+1,p])
        P_t_T_array = np.zeros([q,T+1,p,p])
        P_tt_T_array = np.zeros([q,T+1,p,p])
        # to be implemented by parfor
        for r in range(q):
            Q0 = Gamma0.dot(Gamma0.T)
            tmp_posterior = Kalman_smoothing(T,Q0,A,Q,GL,R,y_array[r])
            u_t_T_array[r] = tmp_posterior['u_t_T']
            P_t_T_array[r] = tmp_posterior['P_t_T']
            P_tt_T_array[r] = tmp_posterior['P_tt_T']
        B_mat = get_posterior_matrix(u_t_T_array, P_t_T_array, P_tt_T_array, y_array)
        # M step        
        A_old = A.copy()
        Gamma_old = Gamma.copy()
        sigma_J_list_old = sigma_J_list.copy()
        L_list_old = copy.deepcopy(L_list)
        Gamma0_old = Gamma0.copy()
        obj_old = obj
        tmp_result = update_param_M_step(B_mat, G, ROI_list, Sigma_E,
                        Gamma0, A, Gamma, sigma_J_list, L_list,
                        flag_A_time_vary = flag_A_time_vary,
                        prior_Q0 = prior_Q0, prior_A = prior_A, prior_Q = prior_Q,
                        prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,
                        MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0,
                        MaxIter = MaxIter, tol = tol, verbose = verbose, L_flag = L_flag)
        Gamma0, A, Gamma = tmp_result['Gamma0'], tmp_result['A'], tmp_result['Gamma'] 
        sigma_J_list, L_list = tmp_result['sigma_J_list'], tmp_result['L_list']
        obj = tmp_result['obj_Mstep']
        # entropy of the posterior too hard to compute
        #===========================================        
        diff_obj = (obj_old-obj)/np.abs(obj_old)         
        diff_param = np.linalg.norm(A-A_old)/np.linalg.norm(A)\
              +np.linalg.norm(Gamma-Gamma_old)/np.linalg.norm(Gamma)\
              +np.linalg.norm(sigma_J_list-sigma_J_list_old)/np.linalg.norm(sigma_J_list)\
              +np.linalg.norm(Gamma0-Gamma0_old)/np.linalg.norm(Gamma0)
        for i in range(p):
            diff_param += np.linalg.norm(L_list[i]-L_list_old[i])/np.linalg.norm(L_list[i])
        if verbose:
            #print "diff A  %f\n , Gamma %f\n, sigma %f\n, Gamm0 %f\n"\
            #  %(np.linalg.norm(A-A_old)/np.linalg.norm(A),
            #  +np.linalg.norm(Gamma-Gamma_old)/np.linalg.norm(Gamma),
            #  +np.linalg.norm(sigma_J_list-sigma_J_list_old)/np.linalg.norm(sigma_J_list),
            #  +np.linalg.norm(Gamma0-Gamma0_old)/np.linalg.norm(Gamma0))
            print "EM Iter %d, diff_param = %f, diff_obj = %f, M-step obj = %f" \
                      %(IterCount, diff_param, diff_obj, obj)
            print "obj"
            print get_neg_llh_y(y_array, Gamma0, A, Gamma, sigma_J_list, L_list,
                   G, ROI_list, Sigma_E, prior_Q0 = prior_Q0, 
                   prior_A = prior_A, prior_Q = prior_Q,
                   prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,  
                   L_flag = L_flag, flag_A_time_vary = flag_A_time_vary, eps = 1E-40)
        IterCount += 1  
                               
    obj_llh = get_neg_llh_y(y_array, Gamma0, A, Gamma, sigma_J_list, L_list,
                   G, ROI_list, Sigma_E, prior_Q0 = prior_Q0, 
                   prior_A = prior_A, prior_Q = prior_Q,
                   prior_L_precision = prior_L_precision, prior_sigma_J_list = prior_sigma_J_list,  
                   L_flag = L_flag, flag_A_time_vary = flag_A_time_vary, eps = 1E-40)

    result = dict(Gamma0 = Gamma0, A= A, Gamma = Gamma, sigma_J_list = sigma_J_list, L_list = L_list,
                  u_t_T_array = u_t_T_array, EM_obj = obj, obj = obj_llh)
    return result

#========== obtain Q0, Q, A if u_array is known ========================
#========== if u_array is known,  compute B1_B3 accordingly
# B1_t = \sum_r u_t u_t^T, B2_t = \sum_r u_t, u_t_1^T, B3_t = \sum_r u_t_1, u_t_1^T
def get_param_given_u(u_array, Gamma0_0, A_0, Gamma_0, 
       flag_A_time_vary = False,
       prior_Q0 = None, prior_A = None, prior_Q = None,
       MaxIter0 = 100, tol0 = 1E-3, verbose0 = True,
       MaxIter = 100, tol = 1E-6, verbose = True):
    """
    u_array, [q,T+1,p]
    """
    q,T0,p = u_array.shape
    T = T0-1
    B1_B3 = np.zeros([T+1,p,p])
    B2 = np.zeros([T,p,p])
    for t in range(T+1):
        B1_B3[t] += u_array[:,t,:].T.dot(u_array[:,t,:])
        if t+1 <= T:
            B2[t] += u_array[:,t+1,:].T.dot(u_array[:,t,:])
    # Q0        
    Q0 = u_array[:,0,:].T.dot(u_array[:,0,:])/np.float(q)
    Gamma0 = np.linalg.cholesky(Q0)        
    # A, Q
    if prior_A is None and prior_Q is None:
        if flag_A_time_vary:
            p = Gamma0_0.shape[0]
            A = np.zeros([T,p,p])
            for t in range(T):
                A[t] = B2[t].dot( np.linalg.inv(B1_B3[t])) 
        else:
            A = B2.sum(axis = 0).dot( np.linalg.inv( (B1_B3[0:T].sum(axis = 0)) ))
        # update Q    
        tmp = _util_obj_grad_AQ_sum_B123(A, B1_B3, B2, T, flag_A_time_vary)
        Q = tmp/np.float(q)/np.float(T)
        Gamma = np.linalg.cholesky(Q)
    else:  # coordinate descent
        A, Gamma, obj = coor_descent_AQ(A_0, Gamma_0, B1_B3, B2, T,q, prior_A, prior_Q,
                    flag_A_time_vary, A_flag = True, Q_flag = True,
                      tau = 0.8, step_ini = 1.0, MaxIter = MaxIter, tol = tol,
                      verbose = verbose,
                      MaxIter0 = MaxIter0, tol0 = tol0, verbose0 = verbose0)
    return Gamma0, A, Gamma                  

# ============= spectral methods to estimate A and C given y_array
# https://www.cs.cmu.edu/~ggordon/spectral-learning/boots-slides.pdf
# in my case, if L is fixed, C = GL is known
# steps: estimate \Sigma_k = E[y_t+k, y_t^T], k = 1,2
# U = left n isngular vectors of \Sigma_1
# \hat A = U^T \Sigma_2 pinv(U^T \Sigma_1)  = S A S^{-1}
#  A = S^{-1} \hat A S
# \hat C = U \hat{A}^{-1} = C S^{-1}
# if I know C, I can solve S and A
def spectral(y_array, C): 
    """
    DOES NOT WORK!!!!
    """
    q,T0,n = y_array.shape
    p = C.shape[1]
    T = T0-1
    A = np.zeros([T,p,p])
    A_hat = np.zeros([T,p,p])
    for t in range(0,T-1):
        Sigma_1 = (y_array[:,t+1,:].T.dot(y_array[:,t,:]))/np.float(q)
        Sigma_2 = (y_array[:,t+2,:].T.dot(y_array[:,t,:]))/np.float(q)
        W,_,_ = np.linalg.svd(Sigma_1, full_matrices = False)
        W = W[:,0:p]
        WT_Sigma_1 = W.T.dot(Sigma_1)
        WT_Sigma_2 = W.T.dot(Sigma_2)
        tmp_A_hat = WT_Sigma_2.dot(np.linalg.pinv(WT_Sigma_1))
        WTC = W.T.dot(C)
        A[t+1] = reduce(np.dot, [np.linalg.inv(WTC), tmp_A_hat, WTC])
        A_hat[t+1] = tmp_A_hat
        
        ''' # debug
            V = u_array[:,t,:].T.dot(u_array[:,t,:])/np.float(q)
            Sigma1_true = reduce(np.dot, [C, A_true[t],V,C.T])
            Sigma2_true = reduce(np.dot, [C, A_true[t+1], A_true[t], V,C.T])
            W_true,_,_ = np.linalg.svd(Sigma1_true, full_matrices = False)
            W_true = W_true[:,0:p]
            WTC_true = W.T.dot(C)
            true_tmp_A_hat = reduce(np.dot, [WTC_true, A_true[t+1], np.linalg.inv(WTC_true)])
            to_show = [Sigma_1, Sigma1_true, Sigma_1-Sigma1_true,
                       Sigma_2, Sigma2_true, Sigma_2-Sigma2_true,
                       tmp_A_hat, true_tmp_A_hat, tmp_A_hat-true_tmp_A_hat]
            plt.figure();
            for i0 in range(9):
                _=plt.subplot(3,3,i0+1); 
                _=plt.imshow(to_show[i0], interpolation = "none"); _=plt.colorbar()
            #Sigma_1, Sigma_2 = Sigma1_true, Sigma2_true
        '''
    return A
     
#=================== compute the -log(y) ======================================
#===== compute marginal covariance of U =======================================
def get_cov_u(Q0, A, Q, T, flag_A_time_vary = False):
    """
    Inputs: Q0, [p,p]; A[T,p,p] or [p,p]; Q [p,p]; T, int; 
    Output: cov_u, [(T+1)p, (T+1)p]
    
    Testing:
    p = 2; T = 5; Q0 = np.eye(p);
    A = np.random.randn(T,p,p); Q = np.eye(p); flag_A_time_vary = True
    A = np.random.randn(p,p);  flag_A_time_vary = False;
    n = 3; R = np.eye(n); C = np.random.randn(n,p); 
    n_trial = 10000;
    u_array, y_array = simulate_kalman_filter_data(T, Q0, A, Q, C, R, n_trial)
    hat_cov_u = np.zeros([(T+1)*p,(T+1)*p])
    for i in range(T+1):
        for j in range(T+1):
            if i == j:
                hat_cov_u[i*p:(i+1)*p, j*p:(j+1)*p] = np.cov(u_array[:,i,:].T)
            else:
                tmp = np.vstack([u_array[:,i,:].T, u_array[:,j,:].T])
                tmp_cov = np.cov(tmp)
                hat_cov_u[i*p:(i+1)*p, j*p:(j+1)*p] = tmp_cov[0:p, p:]
                hat_cov_u[j*p:(j+1)*p, i*p:(i+1)*p] = hat_cov_u[i*p:(i+1)*p, j*p:(j+1)*p].T.copy()
    # another way of computing hat_cov_u, verified
    u_reshape = u_array.reshape([-1,(T+1)*p]); hat_cov_u1 = np.cov(u_reshape.T)
    y_reshape = y_array.reshape([-1,(T+1)*n]); hat_cov_y = np.cov(y_reshape.T)
    
    cov_u = get_cov_u(Q0, A, Q, T, flag_A_time_vary = flag_A_time_vary)
    print np.linalg.norm(cov_u -hat_cov_u)/np.linalg.norm(hat_cov_u)
    #print np.linalg.norm(cov_y -hat_cov_y)/np.linalg.norm(hat_cov_y)
    
    import matplotlib.pyplot as plt; plt.figure();
    #to_plot = list([hat_cov_u, cov_u, cov_u-hat_cov_u])
    to_plot = list([hat_cov_y, cov_y, cov_y-hat_cov_y])
    for i in range(3):
        _=plt.subplot(1,3,i+1); _=plt.imshow(to_plot[i], interpolation = "none")
        _=plt.colorbar()
"""    
    p = Q0.shape[0]
    if flag_A_time_vary == False:
        A0 = A.copy()
        A = np.zeros([T,p,p])
        for t in range(T):
            A[t] = A0.copy()
            
    tilde_A = np.zeros([T,T],dtype = np.object)       
    for i in range(T):
        tilde_A[i,i] = A[i].copy()
    for i in range(T):
        for j in range(i+1,T):
            tmp = np.eye(p)
            for l in range(j,i-1,-1):
                tmp = (np.dot(tmp, A[l])).copy()
            tilde_A[i,j] = tmp
    # once tilde_A is created, it should be read-only
    cov_u_t = np.zeros([T+1,p,p])
    cov_u_t[0] = Q0.copy()
    for t in range(T):
        # cov(u_t)
        tmp = reduce(np.dot, [tilde_A[0,t], Q0, tilde_A[0,t].T])
        tmp += Q
        for i in range(1,t+1):
            tmp += reduce(np.dot, [tilde_A[t-i+1,t], Q, tilde_A[t-i+1,t].T])
        cov_u_t[t+1] = tmp.copy()
    cov_u = np.zeros([p*(T+1), p*(T+1)])
    for i in range(T+1):
        for j in range(i,T+1):
            if i == j:
                cov_u[i*p:(i+1)*p, j*p:(j+1)*p] = cov_u_t[i]      
            else:
                cov_u[i*p:(i+1)*p, j*p:(j+1)*p] = reduce(np.dot, [cov_u_t[i], tilde_A[i,j-1].T])
                cov_u[j*p:(j+1)*p, i*p:(i+1)*p] = cov_u[i*p:(i+1)*p, j*p:(j+1)*p].T.copy()
    return cov_u 
        
#======neg_llh of y
def get_neg_llh_y(y_array, Gamma0, A, Gamma, sigma_J_list, L_list,
                   G, ROI_list, Sigma_E, prior_Q0 = None, 
                   prior_A = None, prior_Q = None,
                   prior_L_precision = None, prior_sigma_J_list = None,  
                   L_flag = False,
                   flag_A_time_vary = True, eps = 1E-40):
    """ priors to be added """                  
    
    # compute R and C first
    GL = np.zeros([G.shape[0],len(L_list)])
    for l in range(len(L_list)):
        GL[:,l] = G[:,ROI_list[l]].dot(L_list[l])    
    C = GL
    R = Sigma_E.copy() 
    for l in range(len(sigma_J_list)):
        R += sigma_J_list[l]**2 *  G[:, ROI_list[l]].dot(G[:, ROI_list[l]].T) 
        
    q, T0, n = y_array.shape
    T = T0-1
    p = Gamma0.shape[0]
    Q0 = Gamma0.dot(Gamma0.T)
    Q = Gamma.dot(Gamma.T)
    cov_u = get_cov_u(Q0, A, Q, T, flag_A_time_vary = flag_A_time_vary)
    if n<= p: # compute directly cov_y = tilde_C cov_u tilde_C^T
        cov_y = np.zeros([n*(T+1), n*(T+1)])
        for i in range(T+1):
            for j in range(i,(T+1)):
                tmp = cov_u[i*p:(i+1)*p, j*p:(j+1)*p]
                CtmpC = reduce(np.dot, [C, tmp, C.T])
                cov_y[i*n:(i+1)*n, j*n:(j+1)*n] = CtmpC
                if i == j:
                    cov_y[i*n:(i+1)*n, j*n:(j+1)*n] += R
                else:
                    cov_y[j*n:(j+1)*n, i*n:(i+1)*n] = CtmpC.T                    
        inv_cov_y, log_det_cov_y, eigs_cov_y = _util_cov_inv_det(cov_y, eps = eps)
        # debug
        # inv_cov_y1, log_det_cov_y1, eigs_cov_y1 = inv_cov_y.copy(), log_det_cov_y, eigs_cov_y.copy()
    else: # use matrix inversion lemma to obtain log_det_cov_y and inv_cov_y
        # compute (cov_u^{-1} + tilde_C^T tilde_R^{-1} tilde_C)
        inv_cov_u, log_det_cov_u, eigs_cov_u = _util_cov_inv_det(cov_u, eps = eps)
        # compute inv R
        inv_R, log_det_R, _ = _util_cov_inv_det(R, eps = eps)
        C_R_inv_C = reduce(np.dot, [C.T, inv_R, C])
        inv_cov_u_C_R_inv_C = inv_cov_u.copy()
        for t in range(T+1):
            inv_cov_u_C_R_inv_C[t*p:(t+1)*p,t*p:(t+1)*p] += C_R_inv_C
        # compute the inverse of inv_cov_u_CRC
        inv_inv_cov_u_C_R_inv_C, log_det_inv_cov_C_R_inv_C, _ = \
                _util_cov_inv_det(inv_cov_u_C_R_inv_C, eps = eps)
        log_det_cov_y = log_det_inv_cov_C_R_inv_C + log_det_cov_u + log_det_R*np.float(T+1)
        # compute R^{-1} - R^{-1} C()^{-1} C R^{-1}
        inv_R_C = inv_R.dot(C)
        inv_cov_y= np.zeros([n*(T+1),n*(T+1)])
        for i in range(T+1):
            for j in range(i,(T+1)):
                tmp = inv_inv_cov_u_C_R_inv_C[i*p:(i+1)*p, j*p:(j+1)*p]
                #  R^{-1} tilde C ()^{-1} tilde_C^T R^{-1}
                inv_R_C_tmp_C_inv_R = reduce(np.dot, [inv_R_C, tmp, inv_R_C.T])
                inv_cov_y[i*n:(i+1)*n, j*n:(j+1)*n] = -inv_R_C_tmp_C_inv_R
                if i == j:
                    inv_cov_y[i*n:(i+1)*n, j*n:(j+1)*n] += inv_R
                else:
                    inv_cov_y[j*n:(j+1)*n, i*n:(i+1)*n] = -inv_R_C_tmp_C_inv_R.T                  
    n_llh = q*log_det_cov_y
    for r in range(q):
        tmp_vec_y = y_array[r].ravel(order = 'C')
        n_llh += (inv_cov_y.dot(tmp_vec_y)).dot(tmp_vec_y)
    # priors
    if prior_sigma_J_list is not None:
        # debug
        #print "prior_sigma_J_list"
        n_llh += obj_prior_sigma_J_list(sigma_J_list, prior_sigma_J_list)
    if prior_L_precision is not None and L_flag:
        #print "prior_L"
        # 1/2 in the llh was all dropped
        for i in range(len(L_list)):
            n_llh += np.dot(prior_L_precision[i], L_list[i]).T.dot(L_list[i])
    if prior_A is not None:
        #print "prior_A"
        if flag_A_time_vary:
            n_llh +=  prior_A['lambda0']* np.sum(A**2)
            for t in range(T-1):
                n_llh += prior_A['lambda1']*np.sum((A[t+1]-A[t])**2)     
        else:
            n_llh +=  prior_A['lambda0']* np.sum(A**2)
    # if prior_Q is not None, add the prior       
    if prior_Q is not None:
        #print "prior_Q"
        n_llh += obj_prior_Q(Gamma, prior_Q)
    if prior_Q0 is not None:
        #print "prior_Q0"
        n_llh += obj_prior_Q(Gamma0, prior_Q0)    
    return n_llh

#=========== for initilization, least square solution for U ==================
def get_lsq_u(y_array, R,C, lambda2 = 1.0):
    # y |u  \sim Cu, solve for u
    # y_array, q, T+1, n , # lambda2, penalization
    n,p = C.shape
    Q= np.eye(p)/lambda2 
    QCT = Q.dot(C.T)
    operator = QCT.dot(np.linalg.inv(reduce(np.dot, [C, Q, C.T])+R)) # p times n
    u_array_hat = np.dot(operator, y_array.transpose([0,2,1])).transpose([1,2,0])
    return u_array_hat
    
