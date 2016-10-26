# -*- coding: utf-8 -*-
import numpy as np
import copy


#========= sampling function for Kronecker product covariance
# p(X, | M, U, V) \propto exp( -1/2 tr [V^{-1} (X-M)^T U^{-1} (X-M)])
# see wikipedia "matrix normal distribution"
def sample_kron_cov(R, C, n_sample, mean = None):
    """
    X is vectorized by columns,  the cov of vec(X) is R \kron C
    Input:  R, [p,p] covariance of rows, each row has p entries 
            C, [n,n] covariance of columns
            n_sample, number of samples
            mean, [n, p], the mean parameter, 0 if None.
    Return: X, [n_sample, n, p]
    """
    p = R.shape[0]
    n = C.shape[0]
    X0 = np.random.randn(n_sample, n, p)
    R0 = np.linalg.cholesky(R).T   #R = R0^T R0
    C0 = np.linalg.cholesky(C)     #C = C0 C0^T
    # np.dot can do the matrix product without for loops
    X = np.transpose(np.dot(np.transpose(np.dot(X0, R0), [0,2,1]), C0.T),[0,2,1])
    #== use for loop to verify the line above:
    #X1 = X0.copy()
    #for i in range(n_sample):
    #    X1[i] = (C0.dot(X0[i])).dot(R0)
    if mean is not None:
        X+= mean
    return X    
# Note the Kronecker is non-identifiable up the the transform R/s, s*C
# So be careful for extreme cases where s is extremely large
# will it help when there is an inverse wishart prior on both the spatial and temporal cov?  
# add my prior
#=============== MLE for Kronecker product covariance, given multiple trials of spatio-temporal data===
def get_mle_kron_cov(U, tol = 1E-3, MaxIter = 100):
    """
    Maximum liklihood estimate of the spatio-temporal cov
    Input:  U, [q, p, T] data matrix, already demeaned
    Output: Tcov, [T,T], temporal covariance matrix
            Qcov, [p,p], spatial covariance matrix
    """
    q,p,T = U.shape
    # TBA
    Tcov = np.eye(T)
    Qcov = np.eye(p)
    rel_diff_param = np.inf
    IterCount = 0
    while rel_diff_param >= tol:
        if IterCount >= MaxIter:
            print "MaxIter reached."
        
        old_Qcov = Qcov.copy()
        tmp0 = U.dot(np.linalg.inv(Tcov))
        Qcov = np.zeros([p,p])
        for r in range(q):
            Qcov += np.dot(tmp0[r], U[r].T)
        Qcov /= (q*T)
        
        old_Tcov = Tcov.copy()
        tmp1 = U.transpose([0,2,1]).dot(np.linalg.inv(Qcov))
        Tcov = np.zeros([T,T])
        for r in range(q):
            Tcov += np.dot(tmp1[r], U[r])
        Tcov /= (q*p)
        
        rel_diff_param = (np.sum((Tcov-old_Tcov)**2) + np.sum((Qcov-old_Qcov)**2))\
                    /(np.sum(Tcov**2) + np.sum(Qcov**2))
        IterCount += 1
    
    return Tcov, Qcov
        

#=====================  neg log llh======================
def get_neg_llh_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha = 1.0, beta = 1.0, # prior params
                     prior_Q = False, prior_Sigma_J = False, prior_L = False, prior_Tcov = False, # prior flags
                     ):
    """
    Sensor cov is assumed to be I (temporally independent, spatially pre-whitened)
    Inputs:
       #=========== unknown variables ============
       Phi: [p,p] Cholesky decomposition of Qu, Qu = Phi Phi^T
       sigma_J_list: [n_ROI,1], the standard deviation of the variance  
                     in each ROI, to get the variance, take squares;
                     Sigma_J is a diangonal matrix with blocks of identities 
                     multipled by sigma_J_list**2;
                     n_ROI = p or n_ROI = p+1 depending on whether there is 
                     a null ROI;      
                     (lower case sigma, means \sigma, not the square)
       L_list: [p], list, each cell is a vector of the L's
       T0: [T,T], Cholesky decomposition of the temporal cov Tcov, Tcov (or \bm{T} in the report) = T0 T0^T
       #========== given parameters ================
       ROI_list: [n_ROI], each element is the index list of the ROI, last one 
                   can be for the null ROI
                   #make sure that the ROIs do not overlap at all
       G:   [n,m], forard matrix
       M,   [q, n, T], tensor of the demeaned sensor data, q trials, n sensors, T time points
       q:   number of trials
       # =================prior paramters=======================
       nu: >= p+1, parameter in the inverse Wishart prior of Qu
       V: [p,p], parameter in the inverse Wishart Prior of Qu
       nu: >= T+1, parameter in the inverse Wishart Prior of Tcov
       V1: [T,T], parameter in the inverse Wishart Prior of Tcov
       inv_Q_L_list, [n_ROI_valid] list, each one is a inverse of the prior covariance matrix of L
       alpha, beta: inverse gamma prior for Sigma_J_list
       # ================ prior flags, boolean ===============
       prior_Q: if True, use the inverse Wishart prior for Qu
       prior_Sigma_J: if True, use the inverse gamma prior for sigma_i in Sigma_J
       prior_L: if True, use the Gaussian prior for L
       prior_Tcov: if True, use the inverse Wishart prior for Tcov
    Output:
       neg_llh    
    """              
    # p = n_ROI_valid
    p = Phi.shape[0] 
    n,T = M[0].shape
    n_ROI = len(sigma_J_list)
    #====== obtain SVD of Qbar=====
    Qu = Phi.dot(Phi.T)
    # compute G_Sigma_GT
    G_Sigma_G = np.zeros([n,n])
    for i in range(n_ROI):
        G_Sigma_G += sigma_J_list[i]**2 * np.dot(G[:,ROI_list[i]], G[:,ROI_list[i]].T)
    # compute GL
    GL = np.zeros([n,p])
    # compute each column seperately
    for i in range(p):
        GL[:,i] = G[:,ROI_list[i]].dot(L_list[i])
    Qbar = G_Sigma_G + GL.dot(Qu).dot(GL.T) 
    V_Q, D_Q, V_QT = np.linalg.svd(Qbar)
    V_Q = (V_Q + V_QT.T)/2
    #======== obtain SVD of Tcov =====
    Tcov = T0.dot(T0.T)
    V_T, D_T, V_TT = np.linalg.svd(Tcov)
    V_T = (V_T+V_TT.T)/2
    #========= evaluate log det ( I + Tcov \kron Qbar)
    DQDT = np.outer(D_Q, D_T) # [n,T]
    logdet = np.sum(np.log(DQDT+1.0))
    #========= evaluate the quadratic form =======
    # compute M1 = V_Q.T M[r] V_T = ((M[r] V_T).T V_Q).T
    M1 = ((np.dot(M, V_T).transpose([0,2,1])).dot(V_Q)).transpose([0,2,1])
    quad = np.sum(np.sum(M1**2, axis = 0)/ (DQDT+1.0))
    #======== add the two terms ======
    nllh = q*logdet+quad 
    #=========add proirs ============
    if prior_Q:
        inv_Q = np.linalg.inv(Qu)
        #det_Q = np.linalg.det(Qu)
        log_det_Q = np.sum(np.log(np.diag(Phi)**2))
        nllh +=  np.float(nu+p+1)*log_det_Q+  np.sum(V*inv_Q) #np.trace(V.dot(inv_Q))
    if prior_Sigma_J:
        # 1/2 in the llh was all dropped
        for i in range(n_ROI):
            nllh += (alpha+1.0)*np.log(np.abs(sigma_J_list[i])) + 0.5*beta/(sigma_J_list[i]**2)
    if prior_L:
        # 1/2 in the llh was all dropped
        for i in range(p):
            nllh += np.dot(inv_Q_L_list[i], L_list[i]).T.dot(L_list[i])
    if prior_Tcov:
        inv_Tcov = (V_T/D_T).dot(V_T.T)
        log_det_Tcov = np.sum(np.log(D_T))
        # np.trace (V1.dot (inv_Tcov)   = sum(V1 * inv_Tcov), since V1 and inv_Tcov are symmetric, 
        # and the trace here is matrix inner product
        nllh +=  np.float(nu1+T+1)*log_det_Tcov+ np.sum(V1*inv_Tcov)   
    return nllh
  
#============== utility functions to compute gradients, T0====================  
def get_dTcov_dT0(T0, h, l):
    """
    Compute the gradient matrix of Tcov = T0 T0^T with respect to T0[h,l]
    Note h>= l, but I will not spend time to check it in this function
    Input: T0, [T, T]  lower triangular matrix
    Output: dTcov, [T,T]
    """    
    T = T0.shape[0]
    dTcov = np.zeros([T,T])
    for i in range(T):
        for j in range(i,T):
            if j!=i:
                if h==i and l<= min(i,j):
                    dTcov[i,j] = T0[j,l]
                elif h==j and l<= min(i,j):
                    dTcov[i,j] = T0[i,l]
                dTcov[j,i] = dTcov[i,j]
            else:
                if h==i and l<= min(i,j):
                    dTcov[i,j] = 2.0*T0[i,l]
    return dTcov  
#============== utility functions to compute gradients, Gamma (or Phi)==================
def get_dQbar_dPhi(A, APhi, s, w):
    """
    Compute the gradient matrix of Qbar =G_Sigma_G + GL.dot(Qu).dot(GL.T)
    with respect to Phi[s,w] (Gamma in the text is Phi here)
    Note s>= w, but I will not spend time to check it in this function
    Input: A = GL, [n, p]
           APhi, [n, p] A.dot(Phi) 
    Output: dQbar [n, n]
    """ 
    tmp = np.outer(A[:,s], APhi[:,w])
    dQbar = tmp + tmp.T
    return dQbar

#============== utility functions to compute gradients, L==================
def get_dQbar_dL(G, GLQu, s, w):
    """
    Compute the gradient matrix of Qbar =G_Sigma_G + GL.dot(Qu).dot(GL.T)
    with respect to L[s,w], note this entry needs to be in an ROI, but I will
    not check it here. 
    Input: G, [n, m]
           GLQu, [n,p] 
    Output: dQbar [n, n]
    """    
    tmp = np.outer(G[:,s], GLQu[:,w])
    dQbar = tmp + tmp.T
    return dQbar 
    
#============== utility functions to compute gradients, sigma_J==================
def get_dQbar_dsigma_J(G, ind, sigma_J_input):
    """
    Compute the gradient matrix of Qbar =G_Sigma_G + GL.dot(Qu).dot(GL.T)
    with respect to a particular ROI, ind are the indices in the ROI
    Input: G, [n, m]
           ind,  indices of the source points in the ROI
           sigma_J_input, the one sigma of the current ROI
    Output: dQbar [n, n]
    """    
    tmp = G[:,ind]
    tmpGGT = np.dot(tmp, tmp.T)
    dQbar = 2.0*sigma_J_input*tmpGGT
    return dQbar     
           
#==== utility functions to compute logdet gradients, when dTcov or dQbar w.r.t a parameter is known
def get_grad_dnegllh_dparam(V_T, D_T, V_Q, D_Q, Mtilde, DQDT, q, tmpgrad, isTcov):
    """
    Compute the gradient of the neg_log_llh, onces dTcov/dparam or dQbar/dparam is known,
    where logdet and the quadratic terms are computed seperately. 
    The parameters one single entry in  T0, Gamma, L, or sigma_J_list 
    Inputs:
        V_T: [T,T] SVD of T_cov
        D_T: [T] SVD of T_cov
        V_Q: [n,n] SVD of Qbar
        D_Q: [n] SVD of Qbar
        Mtilde: [q,n,T], the transformed sensor data
        DQDT: [n,T] outer product of D_Q and D_T
        q: number of trials
        tmpgrad: current gradient, either dTcov_dT0, or dQbar_dGamma, dQbar_dsigma, dQbar_dL
                 note that these are all w.r.t one single entry
        isTcov: boolean, if True, tmpgrad is dTcov_dT0, if False, the other three (of dQbar)
    Output: grad = q* grad of logdet + grad of the quadratic form 
    """   
    if isTcov:
        V = V_T
    else: 
        V = V_Q
    # tmp_grad must be a symmetric matrix, 
    # either [T,T] (isTcov = True) or [n,n] (isTcov = False), so we do cholesky decomposition
    V_tmp_V = np.dot(np.dot(V.T, tmpgrad),V)
    # tmp_grad1: logdet
    if isTcov:  
        tmp1 = np.outer(D_Q, np.diag(V_tmp_V))  # make sure the size is [n,T]
    else:
        tmp1 = np.outer(np.diag(V_tmp_V), D_T)
    tmp_grad1 = (tmp1/(DQDT+1.0)).sum()# [n,T]
    # tmp_grad2, quadratic form
    # for the temporal cov it is
    #  sum_{1}^q vec{Mtilde}^T vec{ D_Q Mtilde VTdTcovVT} 
    if isTcov:
        #Mtilde1 = D_Q Mtilde VTdTcovVT = ((Mtilde VTdTcovVT)^T D_Q)^T
        Mtilde1 = ((Mtilde.dot(V_tmp_V)).transpose([0,2,1])*D_Q).transpose([0,2,1])
    else:
        #Mtilde1 = VQdQbarVQ MtildeD_T = (Mtilde  D_T)^T VQdQbarVQ)^T
        Mtilde1 = (((Mtilde*D_T).transpose([0,2,1])).dot(V_tmp_V)).transpose([0,2,1])    
    tmp_grad2 = -np.sum(Mtilde1*Mtilde)
    grad = np.float(q)*tmp_grad1 + tmp_grad2
    return grad

              
#====================== gradients ======================================
def get_neg_llh_grad_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha = 1.0, beta = 1.0, # prior params
                     prior_Q = False, prior_Sigma_J = False, prior_L = False, prior_Tcov = False, # prior flags
                     Q_flag = True, Sigma_J_flag = True, L_flag = True, Tcov_flag = True):
    """
    Inputs: 
    Others are the same as get_neg_llh_kron
    Q_flag,  Sigma_J_flag, L_flag, Tcov_flag = True Boolean, 
    whether to compute the gradient for each variable
    Output: grad_Phi, grad_sigma_J_list, grad_L_list, grad_T0
    """
    p = Phi.shape[0] 
    n,T = M[0].shape
    n_ROI = len(sigma_J_list)
    #====== obtain SVD of Qbar=====
    Qu = Phi.dot(Phi.T)
    # compute G_Sigma_GT
    G_Sigma_G = np.zeros([n,n])
    for i in range(n_ROI):
        G_Sigma_G += sigma_J_list[i]**2 * np.dot(G[:,ROI_list[i]], G[:,ROI_list[i]].T)
    # compute GL
    GL = np.zeros([n,p])
    # compute each column seperately
    for i in range(p):
        GL[:,i] = G[:,ROI_list[i]].dot(L_list[i])
    Qbar = G_Sigma_G + GL.dot(Qu).dot(GL.T) 
    V_Q, D_Q, V_QT = np.linalg.svd(Qbar)
    V_Q = (V_Q + V_QT.T)/2
    #======== obtain SVD of Tcov =====
    Tcov = T0.dot(T0.T)
    V_T, D_T, V_TT = np.linalg.svd(Tcov)
    V_T = (V_T+V_TT.T)/2
    #============== prepare M to compute the gradient of the quadratic term
    DQDT = (np.outer(D_T, D_Q)).T # [n,T]
    M1 = ((np.dot(M, V_T).transpose([0,2,1])).dot(V_Q)).transpose([0,2,1])
    Mtilde = M1/(DQDT+1.0)
    #=============== initialize the gradients
    grad_Phi = np.zeros(Phi.shape)
    grad_sigma_J_list = np.zeros(n_ROI)
    grad_L_list = list()
    for i in range(p):
        grad_L_list.append(np.zeros(L_list[i].size))
    grad_T0 = np.zeros(T0.shape)
    #=============== gradient for T0 =========        
    if Tcov_flag:
        # for each i,j (j<=i), compute V_T^T dTcov V_T, 
        # then compute the gradient for log det and the quadratic form
        for i in range(T):
            for j in range(0,i+1):
                dTcov = get_dTcov_dT0(T0, i, j)
                grad_T0[i,j] = get_grad_dnegllh_dparam(V_T, D_T, V_Q, D_Q, Mtilde, DQDT,q,dTcov, isTcov = True)
        if prior_Tcov:
            invT_cov = np.dot(V_T/D_T, V_T.T)
            grad_T0 += np.tril(2.0* np.dot(invT_cov.dot( (nu1+T+1) *np.eye(T) - V1.dot(invT_cov)), T0))       
    #============== gradient for Phi in Qu======
    if Q_flag: 
        A = GL
        APhi = A.dot(Phi)
        for i in range(p):
            for j in range(0,i+1):
                dQbar_dPhi = get_dQbar_dPhi(A, APhi, i, j)
                grad_Phi[i,j] = get_grad_dnegllh_dparam(V_T, D_T, V_Q, D_Q, Mtilde, DQDT,q, dQbar_dPhi, isTcov = False)
        if prior_Q:
            invQ = np.linalg.inv(Qu)
            grad_Phi += np.tril(2.0* np.dot(invQ.dot( (nu+p+1) *np.eye(p) - V.dot(invQ)), Phi))
    #============== gradient for entries in L_list in Qbar======   
    if L_flag:
        GLQu = np.dot(GL, Qu)
        for i in range(p):
            for j in range(len(ROI_list[i])):
                ind = ROI_list[i][j]
                # the ind's column, the i th row in L
                dQbar_dL = get_dQbar_dL(G, GLQu, ind, i)
                grad_L_list[i][j] = get_grad_dnegllh_dparam(V_T, D_T, V_Q, D_Q, Mtilde, DQDT,q, dQbar_dL, isTcov = False)
            if prior_L:
                grad_L_list[i] += 2.0*inv_Q_L_list[i].dot(L_list[i])   
    #============== gradient for sigma_J_list in Qbar======
    # ROI_list: [n_ROI], each element is the index list of the ROI, last one can be for the null ROI, 
    # so its length can be 1 larger than p
    if Sigma_J_flag:
        for i in range(n_ROI):
            dQbar_dsigma_J = get_dQbar_dsigma_J(G, ROI_list[i], sigma_J_list[i])
            grad_sigma_J_list[i] = get_grad_dnegllh_dparam(V_T, D_T, V_Q, D_Q, Mtilde, DQDT,q, dQbar_dsigma_J, isTcov = False)
            if prior_Sigma_J:
                grad_sigma_J_list[i] += (alpha+1.0)/sigma_J_list[i]-beta/sigma_J_list[i]**3        
    #========== return the final gradients ===================
    return grad_Phi, grad_sigma_J_list, grad_L_list, grad_T0
    
    
    
# =================map, update one or more varibles============================
def get_map_kron(Phi0, sigma_J_list0, L_list0, T00, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha = 1.0, beta = 1.0, # prior params
                     prior_Q = False, prior_Sigma_J = False, prior_L = False, prior_Tcov = False, # prior flags
                     Q_flag = True, Sigma_J_flag = True, L_flag = True, Tcov_flag = True,
                     tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6, verbose = True # optimization params
                     ):
    """
    update one or more varibles
    Inputs:
        tau: shrinking varible for backtracking
        step_ini: initial step for gradient descent
        MaxIter: Maxium number of iteration
        tol: when the change of relative neg_llh is smaller than tol, stop
        verbose: if True, output each iteration
    Outputs:
        Phi, sigma_J_list, L_list, T0, obj
    """                      
    diff_obj = np.inf
    obj = 1E10
    old_obj = 0
    IterCount = 0
    Phi = Phi0.copy()
    sigma_J_list = sigma_J_list0.copy()
    L_list =copy.deepcopy(L_list0)
    T0 = T00.copy()
    p = len(L_list0)
    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break                  
        grad_Phi, grad_sigma_J_list, grad_L_list, grad_T0 =  get_neg_llh_grad_kron(
                     Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                     Q_flag, Sigma_J_flag, L_flag, Tcov_flag)
        f = get_neg_llh_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov) 
        step = step_ini
        tmp_diff = np.inf
        # back tracking
        while tmp_diff > 0:
            step *= tau
            ref = f -step/2.0*(np.sum(grad_Phi**2) + np.sum(grad_sigma_J_list**2) + np.sum(grad_T0**2))
            for i in range(p):
                ref = ref-step/2.0*(np.sum(grad_L_list[i]**2))
            
            tmp_Phi = Phi-step*grad_Phi*np.float(Q_flag)
            tmp_sigma_J_list = sigma_J_list-step*grad_sigma_J_list*np.float(Sigma_J_flag) 
            tmp_L_list = copy.deepcopy(L_list)
            tmp_T0 = T0- step*grad_T0*np.float(Tcov_flag)
            for i in range(p):
                tmp_L_list[i] -= step*grad_L_list[i]*np.float(L_flag) 
            tmp_f =  get_neg_llh_kron(tmp_Phi, tmp_sigma_J_list, tmp_L_list, tmp_T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov) 
            tmp_diff = tmp_f-ref
        
        
        Phi = tmp_Phi.copy()
        sigma_J_list = tmp_sigma_J_list.copy()
        L_list = copy.deepcopy(tmp_L_list)
        T0 = tmp_T0.copy()
        
        old_obj = obj
        obj = tmp_f
        diff_obj = old_obj-obj
        if verbose:
            print "Iter %d" %IterCount
            print "obj = %f" %obj
            print "diff_obj = %f" %diff_obj
        IterCount += 1
    return Phi, sigma_J_list, L_list, T0, obj
#==============================================================================
# update both Qu and Sigma_J
def get_map_coor_descent_kron(Qu0, Sigma_J_list0, L_list0, Tcov0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha = 1.0, beta = 1.0, # prior params
                     prior_Q = False, prior_Sigma_J = False, prior_L = False, prior_Tcov = False, # prior flags
                     Q_flag = True, Sigma_J_flag = True, L_flag = True, Tcov_flag = True,
                     tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6, verbose = True, # optimization params
                     MaxIter0 = 20, tol0 = 1E-4, verbose0 = True):
    """
    Use cooardinate desent for MAP estimation 
    Inputs:
       #=========== unknown variables ============
       Qu0, [p,p] initial value of the ROI covariance, Phi: [p,p] the cholesky 
             decoposition of Qu, Qu = Phi PhiT
       Sigma_J_list0: [n_ROI,1], initial variance in each ROI, (already 
             in the square form)
             sigma_J_list, [n_ROI,1], the corresponding standard deviation, 
             the actual parameter to optimize
             Sigma_J is a diangonal matrix with blocks of identities multipled 
             by sigma_J_list**2; n_ROI = p or n_ROI = p+1 depending on whether 
             there is a null ROI;               
       L_list0: [p], list, each cell is a vector of the initial L's, 
             L_list is the parameter to estimate
       Tcov0: [T,T] initial value of the temporal covariance, 
             T0: [T,T],  Cholesky decomposition of the temporal 
                covariance Tcov, Tcov = T0 T0^T 
       #========== given parameters ================
       ROI_list: [n_ROI], each element is the index list of the ROI, last one 
                   can be for the null ROI
                   #make sure that the ROIs do not overlap at all
       G:   [n,m], forard matrix
       M,  [q,n,T], demeaned sensor time series data of q trials
       q:  number of trials
       # =================prior paramters=======================
       nu: >= p+1, parameter in the inverse Wishart prior of Qu
       V: [p,p], parameter in the inverse Wishart Prior of Qu
       nu: >= T+1, parameter in the inverse Wishart Prior of Tcov
       V1: [T,T], parameter in the inverse Wishart Prior of Tcov
       inv_Q_L_list, [n_ROI_valid] list, each one is a inverse of the prior covariance matrix of L
       alpha, beta: inverse gamma prior for Sigma_J_list
       # ================ prior flags, boolean ===============
       prior_Q: if True, use the inverse Wishart prior for Qu
       prior_Sigma_J: if True, use the inverse gamma prior for sigma_i in Sigma_J
       prior_L: if True, use the Gaussian prior for L
       prior_Tcov: if True, use the inverse Wishart prior for Tcov
       #================= flags, Boolean ======================
       Q_flag,  Sigma_J_flag, L_flag, Tcov_flag = True Boolean,  whether to 
            compute the gradient for each variable
       #==================optimizaiton parameters ============
       tau: shrinking varible for backtracking (inner loop)
       step_ini: initial step for gradient descent (inner loop)
       MaxIter: Maxium number of iteration 
       tol: when the change of relative neg_llh is smaller than tol, stop
       verbose: if True, output each iteration
       MaxIter0, tol0, verbose0, parameter of the inner coordinate descent loop
    Outputs:
       Qu, Sigma_J_list, L_list, Tcov, obj 
    """
    diff_obj = np.inf
    obj = 1E10
    old_obj = 0
    
    IterCount = 0
    Phi = np.linalg.cholesky(Qu0)
    sigma_J_list = np.sqrt(Sigma_J_list0.copy())
    L_list =copy.deepcopy(L_list0)
    T0 = np.linalg.cholesky(Tcov0)

    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break
        old_obj = obj
        # update each variable with the wrapped function
        if Q_flag:
            if verbose:
                print "updating Phi or Qu"
            Phi, sigma_J_list, L_list, T0, obj = get_map_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                     Q_flag = True, Sigma_J_flag = False, L_flag = False, Tcov_flag = False,
                     tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0, verbose = verbose0 # optimization params
                     )        
        if Sigma_J_flag:
            if verbose:
                print "updating Sigma_J"
            Phi, sigma_J_list, L_list, T0, obj = get_map_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                     Q_flag = False, Sigma_J_flag = True, L_flag = False, Tcov_flag = False,
                     tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0, verbose = verbose0 # optimization params
                     )
        if L_flag:
            if verbose:
                print "updating L"
            Phi, sigma_J_list, L_list, T0, obj = get_map_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                     Q_flag = False, Sigma_J_flag = False, L_flag = True, Tcov_flag = False,
                     tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0, verbose = verbose0 # optimization params
                     )  
        if Tcov_flag:
            if verbose:
                print "updating Tcov"
            Phi, sigma_J_list, L_list, T0, obj = get_map_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                     Q_flag = False, Sigma_J_flag = False, L_flag = False, Tcov_flag = True,
                     tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0, verbose = verbose0 # optimization params
                     )  
        diff_obj = old_obj-obj
        if verbose:
            print "coordinate descent"
            print "Iter %d" %IterCount
            print "obj = %f" %obj
            print "diff_obj = %f" %diff_obj
        IterCount += 1
    Qu = Phi.dot(Phi.T)
    Sigma_J_list = sigma_J_list**2
    Tcov = T0.dot(T0.T)
    return Qu, Sigma_J_list, L_list, Tcov, obj    
    
#======================= a simple MNE solution with inverse operator ==========
def get_MNE_inverse_sol(M,G,lambda2):
    """There is too much overheads to call the MNE implementation, so I implement
       a simpler version here.
       According to the MNE tutorial, the inverse operator is 
           R' G^T (G R'G^T + C)^{-1} , R', source cov, C sensor cov,
       We assume C = I here, R' = 1/lambda2, so the inverse operator is
           1/lambda2* G.T ( G 1/lambda2 G^T + I)^1 
         = 1/lambda2 V.T D U.T  (U lambda2/(D^2 +lambda2) U.T) (G = U D V)
         = V.T D/(D^2 + lambda2) U.T   
    """
    U,D,V = np.linalg.svd(G, full_matrices = False)
    inv_op = (V.T *D/(D**2 + lambda2)).dot(U.T)
    inv_sol = (np.dot(M.transpose([0,2,1]), inv_op.T)).transpose([0,2,1])
    return inv_sol
    

#%%
# ================== testing ==================================================
#if False:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m, n, q, T = 500,50,100,20   
    np.random.RandomState()
    r = np.int(np.floor(n*1.0))
    G = (np.random.randn(n,r)* np.random.rand(r)).dot(np.random.randn(r,m))
    normalize_G_flag = True
    if normalize_G_flag:
        G /= np.sqrt(np.sum(G**2,axis = 0))
        
    n_ROI = 5
    #n_ROI_valid = n_ROI-1
    n_ROI_valid = n_ROI
    ROI_list = list()
    n_dipoles = m//n_ROI
    for l in range(n_ROI-1):
        ROI_list.append( np.arange(l*n_dipoles,(l+1)*n_dipoles))
    ROI_list.append( np.arange((n_ROI-1)*n_dipoles,m))

    Q = np.eye(n_ROI_valid)
    Q[0,1], Q[1,0] = 0.4, 0.4
    Q[1,2], Q[2,1] = 0.5, 0.5
    Q = Q*10.0
    
    a0,b0 = 1.0, 1E-1 # a exp (-b ||x-y||^2)
    # Gaussian process kernel for temporal smoothness
    Tcov = np.zeros([T,T])
    for i in range(T):
        for j in range(T):
            Tcov[i,j] = a0 * np.exp(-b0 * (i-j)**2)
    Tcov += 0.2*np.eye(T)
    
    U = sample_kron_cov(Tcov, Q, n_sample = q)
    
    # prior cov for L
    Q_L_list = list()
    for i in range(n_ROI_valid):
        tmp_n = len(ROI_list[i])
        tmp = np.zeros([tmp_n, tmp_n])
        for i0 in range(tmp_n):
            for i1 in range(tmp_n):
                tmp[i0,i1] = a0 * np.exp(-b0 * np.sum((G[:,i0]-G[:,i1])**2))
        Q_L_list.append(tmp)
    inv_Q_L_list = copy.deepcopy(Q_L_list)
    for i in range(n_ROI_valid):
        inv_Q_L_list[i] = np.linalg.inv(Q_L_list[i])
    L_list = list()
    for i in range(n_ROI_valid):
        tmp_n = len(ROI_list[i])
        tmp = np.random.multivariate_normal(np.zeros(tmp_n), Q_L_list[i])
        #tmp = np.ones(tmp_n)
        L_list.append(tmp)
    
    L = np.zeros([m, n_ROI_valid])
    for i in range(n_ROI_valid):
        L[ROI_list[i], i] = L_list[i]
    
    Sigma_J_list = np.ones(n_ROI)
    Sigma_J_list[2] = 2.0
    J = np.zeros([q,m,T])
    for i in range(n_ROI_valid):
        for j in range(len(ROI_list[i])):
            J[:,ROI_list[i][j],:] = L_list[i][j]*U[:,i,:] + (sample_kron_cov(Tcov, np.eye(1)*Sigma_J_list[i], n_sample = q))[:,0,:]
    if n_ROI_valid < n_ROI:
        i = -1
        for j in range(len(ROI_list[i])):
            J[:,ROI_list[i][j],:] = (sample_kron_cov(Tcov, np.eye(1)*Sigma_J_list[i], n_sample = q))[:,0,:]

    E = np.random.randn(q,n,T)
    M = ((J.transpose([0,2,1])).dot(G.T)).transpose([0,2,1]) + E
    M = M- M.mean(axis = 0)
    
    Qu_true, L_list_true, Sigma_J_list_true, Tcov_true = Q.copy(), copy.deepcopy(L_list), Sigma_J_list.copy(), Tcov.copy()
    #================================ solving things =========================      
    #=========== two step MNE=======================
    #Lambda00 =  1.0/Sigma_J_list_true[0]
    Lambda00 = 1.0
    J_two_step = get_MNE_inverse_sol(M,G,Lambda00)
    plt.figure(); plt.plot(J.ravel(), J_two_step.ravel(), '.')
    # estimate the hidden U for each ROI, and then compute teh two-step cov
    U_two_step = np.zeros([q,n_ROI_valid,T])
    sign_align = True
    for i in range(n_ROI_valid):
        J_tmp = J_two_step[:,ROI_list[i],:]
        J_tmp -= J_tmp.mean(axis = 0)
        # method 1
        # align the to the right singular vector for G, then average? 
        # this seemed better?
        if sign_align:
            tmp_G = G[:, ROI_list[i]]
            # tmp_Gv [r,m], first columne being the direction
            tmp_Gu,_,_ = np.linalg.svd(tmp_G, full_matrices = False) 
            tmp_sign = np.sign(np.dot(tmp_G.T, tmp_Gu[:,0]))
            U_two_step[:,i,:] = (J_tmp.transpose([0,2,1])*tmp_sign).mean(axis = 2)
        # method 2
        # SVD of the solution
        else:
            J_tmp_reshape = J_tmp.transpose([0,2,1]).reshape([-1,len(ROI_list[i])])
            J_tmp_u,_,_ = np.linalg.svd(J_tmp_reshape, full_matrices = False)
            U_two_step[:,i,:] = J_tmp_u[:,i].reshape([q,T])
    plt.figure()
    for i in range(n_ROI_valid):
        plt.subplot(1,n_ROI_valid, i+1)
        plt.plot(U_two_step[:,i,:].ravel(), U[:,i,:].ravel(), '.')
    Tcov_two_step, Qu_two_step = get_mle_kron_cov(U_two_step, tol = 1E-6, MaxIter = 100)
    #=========== one-step method =========
    # priors
    nu = n_ROI_valid+1
    V = np.eye(n_ROI_valid)
    nu1 = T+1
    V1 = np.eye(T)
    
    
    # initialize Tcov0 with MLE of M
    #Tcov0 = np.eye(T)
    Tcov0, _ = get_mle_kron_cov(M, tol = 1E-6, MaxIter = 100)
    
    T00 = np.linalg.cholesky(Tcov0)
    tmpPhi = np.random.randn(n_ROI_valid, nu)
    Qu0 = tmpPhi.dot(tmpPhi.T)/nu
    Phi0 = np.linalg.cholesky(Qu0)
    sigma_J_list0 = np.ones(n_ROI)
    Sigma_J_list0 = sigma_J_list0**2
    L_list0 = copy.deepcopy(L_list)
    for i in range(n_ROI_valid):
        #L_list0[i] = np.random.randn(L_list0[i].size)
        L_list0[i] = np.ones(L_list0[i].size)
 
    prior_Q, prior_Sigma_J,prior_L, prior_Tcov = False, False, True, False
    alpha, beta = 1.0, 1.0
                   
    print "initial obj"
    obj0 = get_neg_llh_kron(Phi0, sigma_J_list0, L_list0, T00, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov) # prior flags
    print obj0                 
    print "optimial obj" 
    Phi = np.linalg.cholesky(Qu_true) 
    sigma_J_list = np.sqrt(Sigma_J_list_true)
    T0 = np.linalg.cholesky(Tcov_true)                
    obj_star = get_neg_llh_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov) # prior flags 
    print obj_star
    
    #if True:
        # set some initial value to the truth
        #Phi0 = Phi
        #Qu0 = Phi0.dot(Phi0.T)
        #Sigma_J_list0 = Sigma_J_list_true 
        #L_list0 = L_list_true
        #T00 = T0
        #Tcov0 = T00.dot(T00.T)

    #L_list0 = copy.deepcopy(L_list)
    Qu_hat, Sigma_J_list_hat, L_list_hat, Tcov_hat, obj = get_map_coor_descent_kron(
                     Qu0, Sigma_J_list0, L_list0, Tcov0, # unknown parameters
                     ROI_list, G, M, q, 
                     nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                     prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                     Q_flag = True, Sigma_J_flag = True, L_flag = True, Tcov_flag = True,
                     tau = 0.8, step_ini = 1.0, MaxIter = 10, tol = 1E-6, verbose = True, # optimization params
                     MaxIter0 = 20, tol0 = 1E-4, verbose0 = False)
       
    plt.figure()
    plt.subplot(1,3,1); plt.imshow(Tcov_hat, interpolation = "none"); 
    plt.colorbar(); plt.title('one-step')
    plt.subplot(1,3,2); plt.imshow(Tcov_two_step, interpolation = "none"); 
    plt.colorbar(); plt.title('two-step')
    plt.subplot(1,3,3); plt.imshow(Tcov_true, interpolation = "none"); 
    plt.colorbar(); plt.title("truth")
                                                                             
    # plot the L_list
    plt.figure()
    for i in range(n_ROI_valid):
        plt.subplot(n_ROI_valid,1,i+1); plt.plot(L_list[i],'r')
        plt.plot(L_list_hat[i],'b');plt.legend(['true','estimated']) 
        
    plt.figure()
    for i in range(n_ROI_valid):
        plt.subplot(n_ROI_valid,1,i+1); plt.plot(L_list[i],L_list_hat[i],'.')         

    plt.figure()
    plt.subplot(3,2,1)
    # correlation:
    diag0 = np.sqrt(np.diag(Qu_hat))
    denom = np.outer(diag0, diag0)
    plt.imshow(np.abs(Qu_hat/denom), vmin = 0, vmax = 1, interpolation = "none")
    plt.title('correlation hat');plt.colorbar()
    
   
    plt.subplot(3,2,2)
    plt.imshow(Qu_hat, vmin = None, vmax = None, interpolation ="none")
    plt.title('cov hat');plt.colorbar(); 
        
    
    plt.subplot(3,2,3)
    diag2 = np.sqrt(np.diag(Qu_two_step))
    denom2 = np.outer(diag2, diag2)
    plt.imshow(np.abs(Qu_two_step/denom2), vmin = 0, vmax = 1, interpolation = "none")
    plt.title('correlation two_step');plt.colorbar()
    
    plt.subplot(3,2,4)
    plt.imshow(Qu_two_step, vmin = None, vmax = None, interpolation = "none")
    plt.title('cov_two_step');plt.colorbar()
    

    plt.subplot(3,2,5)
    diag1 = np.sqrt(np.diag(Q))
    denom1 = np.outer(diag1, diag1)
    plt.imshow(np.abs(Q/denom1), vmin = 0, vmax = 1, interpolation = "none")
    plt.title('correlation true'); plt.colorbar()
    
    plt.subplot(3,2,6)
    plt.imshow(Q, vmin = None, vmax = None, interpolation ="none")
    plt.title('cov true');plt.colorbar();   