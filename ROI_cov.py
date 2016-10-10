# -*- coding: utf-8 -*-
"""
# This does not work yet, I definitely need PSD constraints.  Cholesky decomposition?
@author: ying
"""
import numpy as np
import scipy
import time
import copy

#=============================================================================
def get_neg_llh(Phi, sigma_J_list, L_list, ROI_list, G, MMT, q, Sigma_E,
                     nu, V_inv, inv_Q_L_list, alpha = 1.0, beta = 1.0,
                     prior_Q = False, prior_Sigma_J = False, prior_L = False,
                     eps = 1E-13):
    """
    Inputs:
       #=========== unknown variables ============
       Phi: [p,p] the cholesky decoposition of Qu, Qu = Phi PhiT
       sigma_J_list: [n_ROI,1], the standard deviation of the variance 
                     in each ROI, to get the variance, take squares;
                     Sigma_J is a diangonal matrix with blocks of identities 
                     multipled by sigma_J_list**2;
                     n_ROI = p or n_ROI = p+1 depending on whether there is 
                     a null ROI;               
       L_list: [p], list, each cell is a vector of the L's
       #========== given parameters ================
       ROI_list: [n_ROI], each element is the index list of the ROI, last one 
                   can be for the null ROI
                   #make sure that the ROIs do not overlap at all
       G:   [n,m], forard matrix
       MMT, [n,n], M MT, where M is the sensor data of size [n,q]
       q:   number of trials
       Sigma_E: [n,n], sensor covariance matrix, known
       # =================prior paramters=======================
       nu: = p+2, parameter in the Wishart prior of Qu
       V_inv: [p,p], inverse of the parameter V in the Wishart Prior of Qu
       inv_Q_L_list, [n_ROI_valid] list, each one is a inverse of the prior covariance matrix of L
       alpha, beta: inverse gamma prior for Sigma_J_list
       # ================ prior flags, boolean ===============
       prior_Q: use the Wishart prior for Qu
       prior_Sigma_J: use the inverse gamma prior for sigma_i in Sigma_J
       prior_L: use the Gaussian prior for L
    Output:
       neg_llh    
    """                
    # p = n_ROI_valid
    p = Phi.shape[0] 
    n = MMT.shape[0]
    n_ROI = len(sigma_J_list)
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
    cov = Sigma_E + G_Sigma_G + GL.dot(Qu).dot(GL.T) 
    inv_cov = np.linalg.inv(cov)   
    #======use svd to compute eigs, instead of eigvals
    #eigs = np.real(np.linalg.eigvals(cov)) + eps
    _, eigs, _ = np.linalg.svd(cov)
    eigs[eigs <= 0] += eps
    #=====use svd to compute eigs, instead of eigvals
    log_det_cov = np.sum(np.log(eigs))  
    nllh = q*log_det_cov + np.sum(MMT*inv_cov) #np.trace(MMT.dot(inv_cov))
    if prior_Q:
        # Wishart prior
        #det_Q = np.linalg.det(Qu)
        log_det_Q = np.sum(np.log(np.diag(Phi)**2))
        nllh +=  -np.float(nu-p-1)*log_det_Q+ np.sum(V_inv*Qu) 
    if prior_Sigma_J:
        # 1/2 in the llh was all dropped
        for i in range(n_ROI):
            nllh += (alpha+1.0)*np.log(np.abs(sigma_J_list[i])) + 0.5*beta/(sigma_J_list[i]**2)
    if prior_L:
        # 1/2 in the llh was all dropped
        for i in range(p):
            nllh += np.dot(inv_Q_L_list[i], L_list[i]).T.dot(L_list[i])
    return nllh

#==============================================================================
# update both Qu and Sigma_J, gradient of Qu and Sigma J
def get_neg_llh_grad(Phi, sigma_J_list, L_list, ROI_list, G, MMT, q, Sigma_E,
                     nu, V_inv, inv_Q_L_list, alpha = 1.0, beta = 1.0,
                     prior_Q = False, prior_Sigma_J = False, prior_L = False,
                     Q_flag = True, Sigma_J_flag = True, L_flag = True):
    """
    Q_flag,  Sigma_J_flag, L_flag = True Boolean, 
    whether to compute the gradient for each variable
    """

    p = Phi.shape[0]
    n = MMT.shape[0]
    n_ROI = len(sigma_J_list)  
    
    # initialize the gradient 
    grad_Phi = np.zeros(Phi.shape)
    grad_sigma_J_list = np.zeros(n_ROI)
    grad_L_list = list()
    for i in range(p):
        grad_L_list.append(np.zeros(L_list[i].size))
        
    # compute the gradient in terms of Q
    Qu = Phi.dot(Phi.T)
    G_Sigma_G = np.zeros(MMT.shape)
    for i in range(n_ROI):
        G_Sigma_G += sigma_J_list[i]**2 * np.dot(G[:,ROI_list[i]], G[:,ROI_list[i]].T)
    GL = np.zeros([n,p])
    # compute each column seperately
    for i in range(p):
        GL[:,i] = G[:,ROI_list[i]].dot(L_list[i])
    cov = Sigma_E + G_Sigma_G + GL.dot(Qu).dot(GL.T) 
    inv_cov = np.linalg.inv(cov)
    grad_cov = np.dot(inv_cov, (q*np.eye(n) - np.dot(MMT, inv_cov) ))
    
    #====compute the gradients as the flags request==========
    if Q_flag: 
        grad_Phi = 2.0* np.dot( GL.T.dot(grad_cov).dot(GL), Phi)
        if prior_Q:
            invQ = np.linalg.inv(Qu)
            grad_Phi = grad_Phi + 2.0* np.dot( (-invQ*(nu-p-1)+V_inv), Phi)
        grad_Phi = np.tril(grad_Phi)
    if Sigma_J_flag:
        for i in range(n_ROI):
            tmpGGT = np.sum( grad_cov*(np.dot(G[:,ROI_list[i]], G[:,ROI_list[i]].T)))
            grad_sigma_J_list[i] = 2* sigma_J_list[i]*tmpGGT  
            if prior_Sigma_J:
                grad_sigma_J_list[i] += (alpha+1.0)/sigma_J_list[i]-beta/sigma_J_list[i]**3        
    if L_flag:
        tmp = np.dot(grad_cov, GL).dot(Qu)
        grad_L0 = 2.0* np.dot(G.T,tmp)
        for i in range(p):
            grad_L_list[i] = grad_L0[ROI_list[i],i]
            if prior_L:
                grad_L_list[i] += 2.0*inv_Q_L_list[i].dot(L_list[i])   
    return grad_Phi, grad_sigma_J_list, grad_L_list
#================================================================================
# update one or more varibles
def get_map(Phi0, sigma_J_list0, L_list0, ROI_list, G, MMT, q, Sigma_E,
                      nu, V_inv, inv_Q_L_list, alpha = 1.0, beta = 1.0, 
                      prior_Q = False, prior_Sigma_J = False, prior_L = False,
                      Q_flag = True, Sigma_J_flag = True, L_flag = True,
                      tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6,
                      eps = 1E-13, verbose = True):
    """
    update one or more varibles
    """                      
    diff_obj = np.inf
    obj = 1E10
    old_obj = 0
    
    IterCount = 0
    Phi = Phi0.copy()
    sigma_J_list = sigma_J_list0.copy()
    L_list =copy.deepcopy(L_list0)
    
    p = len(L_list0)
    
    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break
                          
        grad_Phi, grad_sigma_J_list, grad_L_list =  get_neg_llh_grad(
                    Phi, sigma_J_list, L_list, ROI_list, G, MMT, q, Sigma_E,
                     nu, V_inv, inv_Q_L_list, alpha, beta,
                     prior_Q, prior_Sigma_J, prior_L,
                     Q_flag, Sigma_J_flag, L_flag)
        f = get_neg_llh(Phi, sigma_J_list, L_list, ROI_list, G, MMT, q, Sigma_E,
                     nu, V_inv, inv_Q_L_list, alpha, beta ,
                     prior_Q, prior_Sigma_J, prior_L, eps)
        step = step_ini
        tmp_diff = np.inf
        while tmp_diff > 0:
            step *= tau
            ref = f -step/2.0*(np.sum(grad_Phi**2) + np.sum(grad_sigma_J_list**2))
            for i in range(p):
                ref = ref-step/2.0*(np.sum(grad_L_list[i]**2))
            
            tmp_Phi = Phi-step*grad_Phi*np.float(Q_flag)
            tmp_sigma_J_list = sigma_J_list-step*grad_sigma_J_list*np.float(Sigma_J_flag) 
            tmp_L_list = copy.deepcopy(L_list)
            for i in range(p):
                tmp_L_list[i] -= step*grad_L_list[i]*np.float(L_flag) 
            
            tmp_f = get_neg_llh(tmp_Phi, tmp_sigma_J_list, tmp_L_list, ROI_list, 
                                G, MMT, q, Sigma_E, nu, V_inv, inv_Q_L_list, alpha, beta,
                                prior_Q, prior_Sigma_J, prior_L, eps)
            tmp_diff = tmp_f-ref
        
        Phi = tmp_Phi.copy()
        sigma_J_list = tmp_sigma_J_list.copy()
        L_list = copy.deepcopy(tmp_L_list)
        old_obj = obj
        obj = tmp_f
        diff_obj = old_obj-obj
        if verbose:
            print "Iter %d" %IterCount
            print "obj = %f" %obj
            print "diff_obj = %f" %diff_obj
        IterCount += 1
    return Phi, sigma_J_list, L_list, obj
#==============================================================================
# update both Qu and Sigma_J
def get_map_coor_descent(Qu0, Sigma_J_list0, L_list0, ROI_list, G, MMT, q, Sigma_E,
                      nu, V_inv, inv_Q_L_list, alpha = 1.0, beta = 1.0, 
                      prior_Q = False, prior_Sigma_J = False, prior_L = False,
                      Q_flag = True, Sigma_J_flag = True, L_flag = True,
                      tau = 0.8, step_ini = 1.0, MaxIter = 100, tol = 1E-6,
                      eps = 1E-13, verbose = True, 
                      MaxIter0 = 20, tol0 = 1E-4, verbose0 = True):
    
    """
    Use cooardinate desent to 
    Inputs:
       #=========== unknown variables ============
       Phi: [p,p] the cholesky decoposition of Qu, Qu = Phi PhiT
       sigma_J_list: [n_ROI,1], the standard deviation of the variance 
                     in each ROI, to get the variance, take squares;
                     Sigma_J is a diangonal matrix with blocks of identities 
                     multipled by sigma_J_list**2;
                     n_ROI = p or n_ROI = p+1 depending on whether there is 
                     a null ROI;               
       L_list: [p], list, each cell is a vector of the L's
       #========== given parameters ================
       ROI_list: [n_ROI], each element is the index list of the ROI, last one 
                   can be for the null ROI
                   #make sure that the ROIs do not overlap at all
       G:   [n,m], forard matrix
       MMT, [n,n], M MT, where M is the sensor data of size [n,q]
       q:   number of trials
       Sigma_E: [n,n], sensor covariance matrix, known
       # =================prior paramters=======================
       nu: >= p+1, parameter in the inverse Wishart prior of Qu
       V_inv: [p,p], inverse of the parameter V in the Wishart Prior of Qu
       Q_L_list, [n_ROI_valid] list, each one is a covariance matrix
       alpha, beta: inverse gamma prior for Sigma_J_list
       # ================ prior flags, Boolean ===============
       prior_Q: use the inverse Wishart prior for Qu
       prior_Sigma_J: use the inverse gamma prior for sigma_i in Sigma_J
       prior_L: use the Gaussian prior for L
       #================= flags, Boolean ======================
       Q_flag,  Sigma_J_flag, L_flag = True Boolean,  whether to compute the gradient for each variable
       
    """
    diff_obj = np.inf
    obj = 1E10
    old_obj = 0
    
    IterCount = 0
    Phi = np.linalg.cholesky(Qu0)
    sigma_J_list = np.sqrt(Sigma_J_list0.copy())
    L_list =copy.deepcopy(L_list0)

    while np.abs(diff_obj/obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break
        old_obj = obj
        # update each variable with the wrapped function
        if Q_flag:
            if verbose:
                print "updating Phi or Qu"
            Phi, sigma_J_list, L_list, obj = get_map(Phi, sigma_J_list, L_list, 
                          ROI_list, G, MMT, q, Sigma_E,
                          nu, V_inv, inv_Q_L_list, alpha, beta, 
                          prior_Q, prior_Sigma_J, prior_L,
                          Q_flag = True, Sigma_J_flag = False, L_flag = False,
                          tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                          eps = eps, verbose = verbose0)
        
        if Sigma_J_flag:
            if verbose:
                print "updating Sigma_J"
            Phi, sigma_J_list, L_list, obj = get_map(Phi, sigma_J_list, L_list, 
                          ROI_list, G, MMT, q, Sigma_E,
                          nu, V_inv, inv_Q_L_list, alpha, beta, 
                          prior_Q, prior_Sigma_J, prior_L,
                          Q_flag = False, Sigma_J_flag = True, L_flag = False,
                          tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                          eps = eps, verbose = verbose0)
        
        if L_flag:
            if verbose:
                print "updating L"
            Phi, sigma_J_list, L_list, obj = get_map(Phi, sigma_J_list, L_list, 
                          ROI_list, G, MMT, q, Sigma_E,
                          nu, V_inv, inv_Q_L_list, alpha, beta, 
                          prior_Q, prior_Sigma_J, prior_L,
                          Q_flag = False, Sigma_J_flag = False, L_flag = True,
                          tau = tau, step_ini = step_ini, MaxIter = MaxIter0, tol = tol0,
                          eps = eps, verbose = verbose0)                    
        
        diff_obj = old_obj-obj
        if verbose:
            print "coordinate descent"
            print "Iter %d" %IterCount
            print "obj = %f" %obj
            print "diff_obj = %f" %diff_obj
        IterCount += 1
    Qu = Phi.dot(Phi.T)
    Sigma_J_list = sigma_J_list**2
    return Qu, Sigma_J_list, L_list, obj
                       
#============= testing example ============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m,n,q = 80,120,100 
    np.random.RandomState()
    r = np.int(np.floor(n*1.0))
    G = (np.random.randn(n,r)* np.random.rand(r)).dot(np.random.randn(r,m))
    #G = np.random.randn(n,m)
    normalize_G_flag = True
    if normalize_G_flag:
        G /= np.sqrt(np.sum(G**2,axis = 0)) 

    n_ROI = 4
    #n_ROI_valid = n_ROI-1
    n_ROI_valid = n_ROI
    ROI_list = list()
    n_dipoles = m//n_ROI
    for l in range(n_ROI-1):
        ROI_list.append( np.arange(l*n_dipoles,(l+1)*n_dipoles))
    ROI_list.append( np.arange((n_ROI-1)*n_dipoles,m))

    Q = np.eye(n_ROI_valid)
    Q[0,1] = 0.4
    Q[1,0] = 0.4
    Q[1,2] = 0.6
    Q[2,1] = 0.6
    Mu = np.ones([n_ROI_valid,1])
    U = np.random.multivariate_normal(Mu[:,0], Q, q).T
    
    a0,b0 = 1.0, 1E-5 # a exp (-b ||x-y||^2)
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
        L_list.append(tmp)
        
    # test the marginal llh, whether the true parameter gives the best results
    L = np.zeros([m, n_ROI_valid])
    for i in range(n_ROI_valid):
        L[ROI_list[i], i] = L_list[i]
    
    Sigma_J_list = np.ones(n_ROI)*0.01
    Sigma_J_list[2] = 0.05
    J = np.random.randn(m,q)
    for i in range(n_ROI_valid):
        J[ROI_list[i],:] = J[ROI_list[i],:]*np.sqrt(Sigma_J_list[i]) +np.outer(L_list[i], U[i,:])
        #varified: np.outer(L_list[i], U[i,:]) =L_list[i][:,np.newaxis].dot(U[i:i+1,:])
    if n_ROI_valid < n_ROI:
        J[ROI_list[-1],:] =  J[ROI_list[-1],:]*np.sqrt(Sigma_J_list[-1])
        
    sigma = 0.1
    Sigma_E = np.eye(n)*sigma
    E = np.random.randn(n,q)*np.sqrt(sigma)
    M = np.dot(G, J) + E
    
    Sigma_J = np.zeros(m)
    for i in range(n_ROI):
        Sigma_J[ROI_list[i]] = Sigma_J_list[i]

    M_demean = (M.T - np.mean(M, axis = 1)).T
    MMT = M_demean.dot(M_demean.T)    
    GL = np.dot(G,L) 
    covM = MMT/q
    cov_ana = Sigma_E + np.dot(G, np.diag(Sigma_J)).dot(G.T) + np.dot(GL, Q).dot(GL.T)
    
 
    #================================ solving things =========================      
    # compute a two step result

    #this is only fair when Sigma_E is proportional to identity
    #lambda2 =  1.0/Sigma_J[0]
    lambda2 = 0.0
    tmp_U,tmp_D,tmp_V = np.linalg.svd(G, full_matrices = False)
    inv_op = (tmp_V.T *tmp_D/(tmp_D**2 + lambda2)).dot(tmp_U.T)
    J_two_step = inv_op.dot(M)
    #J_two_step2 = np.dot(np.linalg.inv(G.T.dot(G) + np.eye(m)*lambda2), G.T.dot(M))

    
    # estimate the hidden U for each ROI, and then compute teh two-step cov
    sign_align = True
    U_two_step = np.zeros([n_ROI_valid, q])
    for i in range(n_ROI_valid):
        J_tmp = J_two_step[ROI_list[i],:]
        #L_tmp = L_list[i]
        #U_two_step[i] = 1.0/np.sum(L_tmp**2)* np.dot(J_tmp.T, L_tmp)
        # use PCA to get it
        J_tmp = (J_tmp.T- np.mean(J_tmp, axis = 1)).T
        if sign_align:
            tmp_G = G[:, ROI_list[i]]
            # tmp_Gv [r,m], first columne being the direction
            tmp_Gu,_,_ = np.linalg.svd(tmp_G, full_matrices = False) 
            tmp_sign = np.sign(np.dot(tmp_G.T, tmp_Gu[:,0]))
            U_two_step[i] = (J_tmp.T*tmp_sign).mean(axis = 1)
        # method 2
        # SVD of the solution
        else:
            tmpu, tmpd, tmpv = np.linalg.svd(J_tmp, full_matrices = 0)
            U_two_step[i] = tmpv[0,:]
    plt.figure()
    for i in range(n_ROI_valid):
        plt.subplot(1,n_ROI_valid, i+1)
        plt.plot(U_two_step[i,:].ravel(), U[i,:].ravel(), '.')
        
    Qu_two_step = np.cov(U_two_step)
    
    # one-step method
    nu = n_ROI_valid+2
    V_inv = np.eye(n_ROI_valid)*1E3
    tmpPhi = np.random.randn(n_ROI_valid, nu)
    Qu0 = tmpPhi.dot(tmpPhi.T)/nu

   
    
    Phi0 = np.linalg.cholesky(Qu0)
    sigma_J_list0 = np.ones(n_ROI)

    eps = 1E-13
    prior_Q, prior_Sigma_J,prior_L = True, False, False
    alpha, beta = 1.0, 1.0
    
    inv_Q_L_list = copy.deepcopy(Q_L_list)
    for i in range(n_ROI_valid):
        inv_Q_L_list[i] = np.linalg.inv(Q_L_list[i])
               
    Sigma_J_list0 = sigma_J_list0**2
     
    # update all parambers
    #Sigma_J_list0 = Sigma_J_list.copy()
    #Qu0 = Q.copy()
    L_list0 = copy.deepcopy(L_list)
    for i in range(n_ROI_valid):
        #L_list0[i] = np.random.randn(L_list0[i].size)
        L_list0[i] = np.ones(L_list0[i].size)
        
    print "initial obj"
    obj0 = get_neg_llh(Phi0, sigma_J_list0, L_list0, ROI_list, G, MMT, q, Sigma_E,
                     nu, V_inv, inv_Q_L_list, alpha, beta,
                     prior_Q, prior_Sigma_J, prior_L, eps)
    print obj0                 
    print "optimial obj" 
    Phi = np.linalg.cholesky(Q) 
    # lower case indicates the square root
    sigma_J_list = np.sqrt(Sigma_J_list)                
    obj_star = get_neg_llh(Phi, sigma_J_list, L_list, ROI_list, G, MMT, q, Sigma_E,
                     nu, V_inv, inv_Q_L_list, alpha, beta,
                     prior_Q, prior_Sigma_J, prior_L, eps)  
    print obj_star
    
    #L_list0 = copy.deepcopy(L_list)
    Qu_hat, Sigma_J_list_hat, L_list_hat, obj = get_map_coor_descent(Qu0, Sigma_J_list0, L_list0,
                      ROI_list, G, MMT, q, Sigma_E,
                      nu, V_inv, inv_Q_L_list, alpha, beta, 
                      prior_Q, prior_Sigma_J, prior_L ,
                      Q_flag = True, Sigma_J_flag = True, L_flag = True,
                      tau = 0.7, step_ini = 1.0, MaxIter = 40, tol = 1E-5,
                      eps = 1E-13, verbose = True, verbose0 = False)    
                                                                              
    
    # plot the L_list
    plt.figure()
    for i in range(n_ROI_valid):
        plt.subplot(n_ROI_valid,1,i+1)
        plt.plot(L_list[i],'r')
        plt.plot(L_list_hat[i],'b')
        plt.legend(['true','estimated'])
    
    plt.figure()
    for i in range(n_ROI_valid):
        plt.subplot(n_ROI_valid,1,i+1)
        plt.plot(L_list[i],L_list_hat[i],'.')

                

    plt.figure()
    plt.subplot(3,2,1)
    # correlation:
    diag0 = np.sqrt(np.diag(Qu_hat))
    denom = np.outer(diag0, diag0)
    plt.imshow(np.abs(Qu_hat/denom), vmin = 0, vmax = 1, interpolation = "none")
    plt.title('correlation hat')
    plt.colorbar()
    
   
    plt.subplot(3,2,2)
    plt.imshow(Qu_hat, vmin = None, vmax = None, interpolation ="none")
    plt.colorbar()
    plt.title('cov hat')
        
    
    plt.subplot(3,2,3)
    diag2 = np.sqrt(np.diag(Qu_two_step))
    denom2 = np.outer(diag2, diag2)
    plt.imshow(np.abs(Qu_two_step/denom2), vmin = 0, vmax = 1, interpolation = "none")
    plt.title('correlation two_step')
    plt.colorbar()
    
    plt.subplot(3,2,4)
    plt.imshow(Qu_two_step, vmin = None, vmax = None, interpolation = "none")
    plt.title('cov_two_step')
    plt.colorbar()
    
    
    plt.subplot(3,2,5)
    diag1 = np.sqrt(np.diag(Q))
    denom1 = np.outer(diag1, diag1)
    plt.imshow(np.abs(Q/denom1), vmin = 0, vmax = 1, interpolation = "none")
    plt.title('correlation true')
    plt.colorbar()
    
    plt.subplot(3,2,6)
    plt.imshow(Q, vmin = None, vmax = None, interpolation ="none")
    plt.colorbar()
    plt.title('cov true')
    