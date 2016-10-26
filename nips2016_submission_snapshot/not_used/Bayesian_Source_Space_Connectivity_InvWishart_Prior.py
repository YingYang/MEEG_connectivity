# -*- coding: utf-8 -*-
"""
# This does not work yet, I definitely need PSD constraints.  Cholesky decomposition?
@author: ying
"""
import numpy as np
import scipy


#=============================================================================
# asssuming ROI mean is zero, and detrend these from the data
# marginal llh with invert Wishart prior W^{-1}(V, \nu) \nu > p-1
# cov = Sigma_E + G(Sigma_J + L Qu L^T ) G^T
# q log(det (cov) ) + tr (M^T M) cov^{-1} + \nu + p+1 log det (Qu) + tr(V Qu^{-1})
def get_neg_log_post(Phi, G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = False):
    """
    Inputs:
        Phi, cholesky decomposition of Qu
        G, [n,m] forward matrix
        MMT, [n,n] M^T M of the sensor data M
        q, number of independent trials of q
        Sigma_E, [n,n] the sensor covariance matrix
        L, [m,p] loading matrix from ROI hidden var to the source dipoles
        Phi, [p,p], lower triangular matrix, Qu = Phi Phi.T (cholesky decomp)
        Sigma_J, [m], the variance at each dipole, the dipoles in the same ROI share the same variance, sigma_i
        nu, integer, degrees of freedom of the inverse wishart, nu > p+1
        V, [p,p] matrix parameter of the inverse wishart.        
    Ouput:
        -log of posterior
        q log(det (cov) ) + tr (M^T M) cov^{-1} + (\nu + p+1) log det (Qu) + tr(V Qu^{-1})
    """
    eps = 1E-13
    p = Phi.shape[0]
    Qu = Phi.dot(Phi.T)
    G_Sigma_G = np.zeros(MMT.shape)
    for i in range(Sigma_J.size):
        G_Sigma_G += Sigma_J[i]* np.outer(G[:,i], G[:,i])
    cov = Sigma_E + G_Sigma_G + GL.dot(Qu).dot(GL.T) 
    inv_cov = np.linalg.inv(cov)
    eigs = np.real(np.linalg.eigvals(cov)) + eps
    log_det_cov = np.sum(np.log(eigs))
    inv_Q = np.linalg.inv(Qu)
    #det_Q = np.linalg.det(Qu)
    log_det_Q = np.sum(np.log(np.diag(Phi)**2))
    if prior_on:
        result = q*log_det_cov + np.trace(MMT.dot(inv_cov)) \
             + np.float(nu+p+1)*log_det_Q+ np.trace(V.dot(inv_Q))
    else:
        result = q*log_det_cov + np.trace(MMT.dot(inv_cov)) 
    return result

#=========================== gradient for Qu only =============================  
def get_neg_log_post_grad_Qu(Phi, G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = False):
    """
        Just get the gradient for Qu, ignore other variables
    """
    p = Phi.shape[0]
    Qu = Phi.dot(Phi.T)
    G_Sigma_G = np.zeros(MMT.shape)
    for i in range(Sigma_J.size):
        G_Sigma_G += Sigma_J[i]* np.outer(G[:,i], G[:,i])
    cov = Sigma_E + G_Sigma_G + GL.dot(Qu).dot(GL.T) 
    inv_cov = np.linalg.inv(cov)
    GLT_inv_cov = np.dot(GL.T, inv_cov)
    invQ = np.linalg.inv(Qu)
    if prior_on:
        grad0 = (q*  GL.T.dot(inv_cov).dot(GL) - GLT_inv_cov.dot(MMT).dot(GLT_inv_cov.T) \
           + invQ.dot( (nu+p+1) *np.eye(p) - V.dot(invQ)))
    else:
        grad0 = q* GL.T.dot(inv_cov).dot(GL) - GLT_inv_cov.dot(MMT).dot(GLT_inv_cov.T)
    grad1 = 2.0* grad0.dot(Phi)
    # cholesky decomposition is lower triangular
    grad = np.tril(grad1)
    return grad

#==============================================================================
# gradient descent optimization, using back track
# only update Qu
def get_map_Qu(Qu0, G, MMT, q, Sigma_E, L, Sigma_J, nu, V, MaxIter = 100, tol = 1E-6,
               beta = 0.8, step_ini = 1.0, eps = 1E-6, prior_on = False):
    diff_obj = np.inf
    obj = np.inf
    old_obj = 1
    IterCount = 0
    Phi = np.linalg.cholesky(Qu0)
    while np.abs(diff_obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break
        GL = np.dot(G, L)
        grad = get_neg_log_post_grad_Qu(Phi, G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = prior_on)
        step = step_ini
        fqu = get_neg_log_post(Phi , G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = prior_on)
        tmp = get_neg_log_post(Phi - step*grad , G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = prior_on)
        while ( np.isnan(tmp) or tmp  > fqu -step/2.0*np.sum(grad**2)):
              step = step*beta
              tmp = get_neg_log_post(Phi - step*grad , G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = prior_on)
        Phi = Phi - step*grad
        old_obj = obj
        #obj = get_neg_log_post(Phi, G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = prior_on)
        obj = tmp
        diff_obj = old_obj-obj
        print "Iter %d" %IterCount
        print "obj = %f" %obj
        print "diff_obj = %f" %diff_obj
        IterCount += 1    
    Qu = Phi.dot(Phi.T)
    return Phi, Qu, obj
#============= testing example ============================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m = 800
    q = 100
    n = 30
    r = np.int(np.floor(n*1.0))
    G = (np.random.randn(n,r)* np.random.rand(r)).dot(np.random.randn(r,m))
    #G = np.random.randn(n,m)
    normalize_G_flag = False
    if normalize_G_flag:
        G /= np.sqrt(np.sum(G**2,axis = 0)) 
    n_ROI = 4
    n_ROI_valid = n_ROI-1
    ROI_list = list()
    n_dipoles = m//4
    ROI_list.append( np.arange(0,n_dipoles))
    ROI_list.append( np.arange(n_dipoles,2*n_dipoles))
    ROI_list.append( np.arange(2*n_dipoles,3*n_dipoles))
    ROI_list.append( np.arange(3*n_dipoles,m))
    
    if True:
        Q = np.array([[1,0,0,-0.2],[0,1,0.5,0],[0,0.5,1,0],[-0.2,0,0,1]])*3
        Mu = np.ones([4,1])
        U = np.random.multivariate_normal(Mu[:,0], Q, q).T
        L_list = list()
        for i in range(n_ROI_valid):
            tmp = np.ones( [len(ROI_list[i]), 1])
            #tmp = np.random.randn(len(ROI_list[i]), 1)
            tmp = tmp/np.linalg.norm(tmp)
            L_list.append( tmp)
        
        # test the marginal llh, whether the true parameter gives the best results
        L = np.zeros([m, n_ROI])
        for i in range(n_ROI_valid):
            L[ROI_list[i], i] = L_list[i][:,0]
    if n_ROI_valid < n_ROI:
        Q = np.array([[1,0,0],[0,1,0.5],[0,0.5,1]])*3
        Mu = np.ones([3,1])
        U = np.random.multivariate_normal(Mu[:,0], Q, q).T
        L_list = list()
        for i in range(n_ROI):
            tmp = np.ones( [len(ROI_list[i]), 1])
            #tmp = np.random.randn(len(ROI_list[i]), 1)
            tmp = tmp/np.linalg.norm(tmp)
            L_list.append( tmp)
        # test the marginal llh, whether the true parameter gives the best results
        L = np.zeros([m, n_ROI-1])
        for i in range(n_ROI-1):
            L[ROI_list[i], i] = L_list[i][:,0]
    
   
    Sigma = np.ones(n_ROI)*0.01
    J = np.random.randn(m,q)
    for i in range(n_ROI_valid):
        J[ROI_list[i],:] = J[ROI_list[i],:]*np.sqrt(Sigma[i]) + L_list[i].dot(U[i:i+1,:])
    
    if n_ROI_valid < n_ROI:
        J[ROI_list[-1],:] =  J[ROI_list[-1],:]*np.sqrt(Sigma[-1])
        
    sigma = 0.1
    Sigma_E = np.eye(n)*sigma
    E = np.random.randn(n,q)*np.sqrt(sigma)
    M = np.dot(G, J) + E
    

    Sigma_J = np.zeros(m)
    for i in range(n_ROI):
        Sigma_J[ROI_list[i]] = Sigma[i]
    
    
    M_demean = (M.T - np.mean(M, axis = 1)).T
    MMT = M_demean.dot(M_demean.T)    
    GL = np.dot(G,L) 
    covM = MMT/q
    cov_ana = Sigma_E + np.dot(G, np.diag(Sigma_J)).dot(G.T) + np.dot(GL, Q).dot(GL.T)
    
    if False:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(covM, interpolation = "none")
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(cov_ana, interpolation = "none")
        plt.colorbar()
        
        print np.linalg.norm(covM-cov_ana)/np.linalg.norm(covM)
        print np.linalg.norm(covM-cov_ana)
        
        plt.figure()
        plt.imshow(cov_ana-covM, interpolation = "none")
    
     
     
    #================================ solving things =========================      
    # compute a two step result
    import sys
    sys.path.insert(0, "/home/ying/Dropbox/MEG_source_loc_proj/source_space_dependence/")
    from One_and_two_step_regressions import two_step_regression
    Lambda00 =  1.0/Sigma_J[0]
    X = np.ones([q,1])
    J_two_step, Beta_two_step= two_step_regression(M, G, X.T, Lambda00, 0.0)
    # estimate the hidden U for each ROI, and then compute teh two-step cov
    U_two_step = np.zeros([n_ROI_valid, q])
    for i in range(n_ROI_valid):
        J_tmp = J_two_step[ROI_list[i],:]
        L_tmp = L_list[i][:,0]
        U_two_step[i] = 1.0/np.sum(L_tmp**2)* np.dot(J_tmp.T, L_tmp)
    
    Qu_two_step = np.cov(U_two_step)
    
    nu = n_ROI_valid+1
    tmpPhi = np.random.randn(n_ROI_valid, nu)
    Qu0 = tmpPhi.dot(tmpPhi.T)/nu
    #Qu0 = np.eye(n_ROI_valid)
    #Qu0 = Q.copy()
   
    #V = Q.copy()
    V = np.eye(n_ROI_valid)
    #V[0,1] = 0.4
    #V[1,0] = 0.4
    
    Phi0 = np.linalg.cholesky(Qu0)
    Phi = np.linalg.cholesky(Q)
    print get_neg_log_post(Phi0, G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = False)
    print get_neg_log_post(Phi, G, MMT, q, Sigma_E, L,Sigma_J, nu, V, GL, prior_on = False)
     
    Phi_hat, Qu_hat, obj = get_map_Qu(Qu0, G, MMT, q, Sigma_E, L, Sigma_J, nu, V, MaxIter = 50, tol = 1E-4,
               beta = 0.8, step_ini = 0.1, prior_on = False)
               
    

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
    