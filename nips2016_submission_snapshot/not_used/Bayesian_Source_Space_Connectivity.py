# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:49:59 2015

@author: ying
"""
import numpy as np
import scipy


#==============================================================================
# marginal llh
def get_margnial_llh(G, M, Sigma_E, L, Qu, Mu, Sigma_J):
    """
    Inputs:
        G, [n,m] forward matrix
        M, [n,q] sensor data 
        Sigma_E, [n,n] the sensor covariance matrix
        L, [m, n_ROI] loading matrix from ROI hidden var to the source dipoles
        Qu, [n_ROI, n_ROI], covariance matrix of the hidden variable u
        Mu, [n_ROI], mean of the  hidden variable u
        Sigma_J, [m], the variance at each dipole, the dipoles in the same ROI share the same variance, sigma_i
    Ouput:
        mllh, proportional to marginal llh
        -log(det( cov) - trace (M - GLMu)^T (M-GLMu)/q (cov)^{-1})
        cov = Sigma_E + G(Sigma_J + L Qu L^T ) G^T
    """
    q = M.shape[1]
    GLMu = np.dot(G, L).dot(Mu)
    demean_M = (M.T - GLMu).T
    MTM = np.dot(demean_M, demean_M.T)/np.float(q)
    cov0 = np.dot(L, Qu).dot(L.T) + np.diag(Sigma_J)
    cov = np.dot(G, cov0).dot(G.T) + Sigma_E
    mllh = -np.log(np.linalg.det(cov)) - np.trace(MTM.dot(np.linalg.inv(cov)))
    return mllh

#===============================================================================
# The optimization is not working! Something is wrong
# notations may be depreciated
def coordinate_descent_opt( G,M,  Sigma_E, ROI_list, J_ini, dim = 1,
                            U_ini = None, Qu_ini = None, L_list_ini = None, Sigma_ini = None,
                            Mu_ini = None,
                            MaxIter = 100, tol = 1E-4, Flag_verbose = False,
                            fix_ROI0 = True):
    """
    The source space noise is assumed to be the same within each ROI. 
    See evernote Aug 6 for the algorithms
    Alternating between all variables to optimize the precition matrix.
    No prior is on Lambda, invQu, Mu.
    Note we assume that the variance of each ROI is the same across source points. 
    This can be relaxed. 

    Parameters:
        G, [n,m] forward matrix
        M, [n,q] sensor data 
        Sigma_E, [n,n] the sensor covariance matrix
        ROI_list, [n_ROI], list of arrays for ROI indices, including the possible null ROI
        J_ini,[m,q], initial values of J
        Sigma_ini,[n_ROI], initial variance for J in each ROI
        dim, integer, the size of low dimensional vector for each ROI, usually only 1, the ROI mean
        L_list_ini, initial L for each ROI, the null ROI can be ignored
        fix_ROI0, boolean, if True, the last ROI is a null ROI, with zero mean, and no variance
            then the dimension of U, Qu is n_ROI_valid = (n_ROI-1)
        
    Output:
        dict of following variables
        U, [n_ROI *dim, q]
        Qu, [n_ROI*dim, n_ROI*dim]
        invQu, [n_ROI*dim, n_ROI*dim], inverse of Qu, precition
        Mu, [n_ROI*n_dim], mean of U
        Sigma, [n_ROI], variance of the dipoles in each ROI
        J, [m,q]
    """
    n,m = G.shape
    n_ROI = len(ROI_list)
    q = M.shape[1]
    if fix_ROI0:
        n_ROI_valid = n_ROI-1
    L_list = np.zeros(n_ROI, dtype = np.object)
    # note that the last one is always the null ROI
    # initialize the mapping
    for i in range(n_ROI):
        tmp_n = len(ROI_list[i])
        L_list[i] = np.zeros([tmp_n, dim])   
    # ==================initialization=======================================
    J = J_ini.copy()    
    if U_ini is None or Qu_ini is None or L_list_ini is None or Sigma_ini is None or Mu_ini is None:
        Sigma = np.zeros(n_ROI)
        # number of effective ROIs
        U = np.zeros([n_ROI_valid*dim, q])
        Qu = np.zeros([n_ROI_valid * dim, n_ROI_valid * dim])
        invQu = np.zeros([n_ROI_valid * dim, n_ROI_valid * dim])
        # for convenience, make Mu two dimensional
        Mu = np.zeros([n_ROI_valid * dim,1])
        # ================further initialization =================================
        # SVD to initialize L, Mu, U, Q, Lambda_i
        for i in range(n_ROI_valid):
            tmp_J = J[ROI_list[i],:]
            tmpu, tmpd, tmpv = np.linalg.svd(tmp_J)
            L_list[i] = tmpu[:,0:dim] # make the weights orthonormal
            U[i*dim: (i+1)*dim,:] = (tmpv[0:dim,:].T* tmpd[0:dim]).T
            tmp_residual = tmp_J - np.dot(L_list[i], U[i*dim: (i+1)*dim,:]) 
            # assume each source point in the ROI is the same!!! 
            # Why? shall we relax this? 
            Sigma[i] = np.var(tmp_residual.ravel())
        if fix_ROI0:
            Sigma[n_ROI-1] = np.var(J[ROI_list[n_ROI-1],:].ravel())
        Mu[:,0] = np.mean(U, axis = 1)
        Qu = np.cov(U, rowvar = 1) # each row is one variable
    else:
        U = U_ini.copy()
        Sigma = Sigma_ini.copy()
        Qu = Qu_ini.copy()
        Mu = Mu_ini.copy()
        for i in range(n_ROI_valid):
            L_list[i] = L_list_ini[i].copy()
        
    invQu = np.linalg.inv(Qu)
    IterCount = 0
    obj = np.inf
    diff_obj = np.inf
    old_obj = 0
   
    # noise covariance matrix
    V_E, d_E, V_ET = np.linalg.svd(Sigma_E)
    V_E = (V_E+ V_ET.T)/2.0
    GTV_E = np.dot(G.T, V_E) #[m,n]
    GTSigma_E_inv_M = np.dot(G.T, (np.dot(V_E/d_E, V_E.T)).dot(M))  
    
    while np.abs(diff_obj) >= tol:
        if IterCount >= MaxIter:
            print "MaxIter achieved"
            break
        
        old_obj = obj
        #======================================================================
        # update Sigma 
        if False:
            for i in range(n_ROI_valid):
                tmp_residual = J[ROI_list[i],:] - np.dot(L_list[i], U[i*dim:(i+1)*dim,:]) 
                # assume each source point in the ROI is the same!!! 
                Sigma[i] = np.mean(tmp_residual**2)
            if fix_ROI0:
                # the null ROI
                Sigma[n_ROI-1] = np.mean(J[ROI_list[-1],:]**2)
            # debug
            print "Sigma"
            print Sigma
        
        #======================================================================
        # update Ui, Li for each ROI
        if True:
            for i in range(n_ROI_valid):
                # update Ui first 
                Li = L_list[i]
                ind_i = np.arange(i*dim,(i+1)*dim)  
                ind_no_i = np.setdiff1d(np.arange(0,n_ROI_valid*dim), ind_i)
                tmp_invQu = invQu[ind_i,:]
                tmp_invQu_i = tmp_invQu[:, ind_i]
                tmp_invQu_no_i = tmp_invQu[:, ind_no_i]
                tmp_Mu_i = Mu[ind_i,:]
                tmp_Mu_no_i = Mu[ind_no_i,:]
                # note the dimension here, [n_dim, q], [n_dim, 1], 
                tmp_U_no_i_demean =U[ind_no_i,:] - tmp_Mu_no_i          
                # denominator, [dim, dim]
                denom = np.dot(Li.T, Li)/Sigma[i] + tmp_invQu_i
                # numerator, sizes of the three terms,  [dim,q], [dim, q], [dim, 1]
                # since U has covariance with other ROIs, there is a cross ROI term here
                nume = 1.0/Sigma[i]* np.dot(Li.T, J[ROI_list[i],:]) \
                         - np.dot(tmp_invQu_no_i, tmp_U_no_i_demean) \
                         + np.dot(tmp_invQu_i,tmp_Mu_i) 
                tmpU = np.dot(np.linalg.inv(denom), nume)
                U[ind_i,:] = tmpU
                # debug do not update this
                # update L, no constraint, applied, will result in unidentifiablity
                if False:
                    # add the constraint that ||L||^2_2 = 1
                    # min (J-LU)^T(J-LU) + \lambda(||L||^2-1)
                    #  L = JU^T/(UU^T+\lambda)
                    tmp =  reduce(np.dot, [J[ROI_list[i]],tmpU.T, np.linalg.inv(tmpU.dot(tmpU.T))])
                    norm_tmp = np.linalg.norm(tmp)
                    if norm_tmp >=1: 
                        L_list[i] = tmp/np.linalg.norm(tmp)
                    else:
                        # the constraint can not be satisfied, what to do?, still normalize it to one?
                        print "the norm 1 constraint can not be satisfied"
                        L_list[i] = tmp/np.linalg.norm(tmp)
       
        #======================================================================
        # update Qu, invQu, Mu
        if True:
            Mu[:,0] = np.mean(U, axis = 1)
        if True:
            U_demean = U-Mu
            Qu = np.dot(U_demean, U_demean.T)/np.float(q)
            if np.linalg.eig(Qu)[0].min() <=0:
                print "COV IS SINGULAR!!!"
                print np.linalg.eig(Qu)[0].min()
                break
            invQu = np.linalg.inv(Qu) 
            
            
        #======================================================================
        # update J
        if True:
            Sigma_diag = np.zeros(m)
            J_mean = np.zeros([m,q])
            for i in range(n_ROI_valid):
                Sigma_diag[ROI_list[i]] = Sigma[i]
                # the group mean for each ROI
                J_mean[ROI_list[i],:] = np.dot(L_list[i], U[i*dim: (i+1)*dim,:])/Sigma[i]
            if fix_ROI0:
                Sigma_diag[ROI_list[-1]] = Sigma[-1]
            prec_J = np.dot(GTV_E/d_E, GTV_E.T) + np.diag(1.0/Sigma_diag)
            # if the m = 7000, it will take forever to do the svd
            u0,d0,v0 = np.linalg.svd(prec_J)   
            u0 = (u0+v0.T)/2.0
            # posterior mean of J, (GT \Sigma_E^{-1} M + Lambda * prior mean
            J = np.dot( np.dot(u0/d0, u0.T), (GTSigma_E_inv_M + J_mean))  
           
        #==================== objective function =======================
        #  obj = negative log likelihood *2 
        # double check this part!!!
        residual_M = np.dot((np.dot(G,J)-M).T, V_E/np.sqrt(d_E))
        obj_JM = np.sum(residual_M**2)
        # the non ROI first
        obj_UJ = np.sum(J[ROI_list[-1],:]**2)/Sigma[-1]
        for i in range(n_ROI_valid):
            residual_J = J[ROI_list[i],:] - np.dot(L_list[i], U[i*dim:(i+1)*dim,:])
            obj_UJ += q* np.log(Sigma[i]) + 1.0/Sigma[i]*np.sum( residual_J**2)
        #print "obj_UJ = %e" %obj_UJ       
        invQu_u, invQu_d, invQu_v = np.linalg.svd(invQu)
        residual_U = np.dot((U - Mu).T, invQu_u* np.sqrt(invQu_d))
        obj_U = -q* np.sum(np.log(invQu_d)) + np.sum(residual_U**2) 
        #print obj_JM, obj_UJ, obj_U
        obj = obj_JM + obj_UJ + obj_U
        diff_obj = (old_obj - obj)/np.abs(obj)
        if Flag_verbose:
            print "iter %d" %IterCount
            print "obj = %e" % obj
            print "diff_obj = %e" % diff_obj
        
        #================compute the the marginal llh ================
        if dim == 1 and Flag_verbose:
            L = np.zeros([m,n_ROI])
            Qu_all = np.zeros([n_ROI,n_ROI])
            Mu_all = np.zeros(n_ROI)
            for l in range(n_ROI):
                L[ROI_list[i],i] = L_list[i][:,0]
            if fix_ROI0:
                Mu_all[0:n_ROI_valid] = Mu[0:n_ROI_valid,0].copy()
                Qu_all[0:n_ROI_valid,0:n_ROI_valid] = Qu.copy()
            else:
                Mu_all = Mu[:,0].copy()
                Qu_all = Qu.copy()   
            mllh = get_margnial_llh(G, M, Sigma_E, L, Qu_all, Mu_all, Sigma_diag)
            print "neg marginal llh = %e" % (-mllh)
        #=================================== 
        IterCount += 1
        
    result = dict(J= J, Mu = Mu, Qu = Qu, invQu = invQu, Sigma = Sigma, U = U, L_list = L_list) 
    return result                  



#============= testing example ============================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m = 40
    q = 100
    n = 41
    
    r = np.floor(n*0.5)
    #G = (np.random.randn(n,r)* np.random.rand(r)).dot(np.random.randn(r,m))
    G = np.random.randn(n,m)
    normalize_G_flag = False
    if normalize_G_flag:
        G /= np.sqrt(np.sum(G**2,axis = 0)) 

    n_ROI = 4
    n_ROI_valid = n_ROI-1
    ROI_list = list()
    ROI_list.append( np.arange(0,10))
    ROI_list.append( np.arange(10,20))
    ROI_list.append( np.arange(20,30))
    ROI_list.append( np.arange(30,m))
    
    Q = np.array([[1,0,0],[0,1,0.5],[0,0.5,1]])
    Mu = np.ones([3,1])
    U = np.random.multivariate_normal(Mu[:,0], Q, q).T
    L_list = list()
    for i in range(n_ROI_valid):
        tmp = np.ones( [len(ROI_list[i]), 1])
        #tmp = np.random.randn(len(ROI_list[i]), 1)
        tmp = tmp/np.linalg.norm(tmp)
        L_list.append( tmp)
    
    Sigma = np.ones(n_ROI)*0.1
    
    J = np.random.randn(m,q)
    for i in range(n_ROI_valid):
        J[ROI_list[i],:] = J[ROI_list[i],:]*Sigma[i] + L_list[i].dot(U[i:i+1,:])
    
    J[ROI_list[-1],:] *= Sigma[-1]
    sigma = 1E-2
    Sigma_E = np.eye(n)*sigma
    E = np.random.randn(n,q)*sigma
    M = np.dot(G, J) + E
    
    
    # test the marginal llh, whether the true parameter gives the best results
    L = np.zeros([m, n_ROI])
    for i in range(n_ROI-1):
        L[ROI_list[i], i] = L_list[i][:,0]
    L[ROI_list[-1], n_ROI-1] = 1.0
    
    Qu = np.zeros([n_ROI, n_ROI])
    Qu[0:n_ROI-1,0:n_ROI-1] = Q.copy()
    Mu_u = np.hstack([Mu[:,0],0] )
    Sigma_J = np.zeros(m)
    for i in range(n_ROI):
        Sigma_J[ROI_list[i]] = Sigma[i]
        
    print get_margnial_llh(G, M, Sigma_E, L, Qu, Mu_u, Sigma_J)
    Sigma_J1 = Sigma_J + np.random.randn(Sigma_J.size)*0.05
    print get_margnial_llh(G, M, Sigma_E, L, Qu, Mu_u, Sigma_J1)
    Qu_1 = np.eye(n_ROI)
    Qu_1[0,1], Qu_1[1,0] = 0.0, 0.0
    Qu_1[2,2]= 0
    print get_margnial_llh(G, M, Sigma_E, L, Qu_1, Mu_u, Sigma_J)
    
    
    # compute a two step result
    import sys
    sys.path.insert(0, "/home/ying/Dropbox/MEG_source_loc_proj/source_space_dependence/")
    from One_and_two_step_regressions import two_step_regression
    Lambda00 = 1E-3
    X = np.ones([q,1])
    J_two_step, Beta_two_step= two_step_regression(M, G, X.T, Lambda00, 0.0)
    

    set_id = 2
    if set_id == 1:
        # set 1, improve the L2 results
        J_ini = J_two_step.copy()
    elif set_id == 2:
        # set 2, use a randoom start
        J_ini = np.random.randn(m,q)
    elif set_id == 3:  
        # set 3, start from the true
        J_ini = J.copy()
   
    
    MaxIter = 100
    tol = 1E-4
    Flag_verbose = True
    dim = 1
    
    Mu_ini = np.random.randn(n_ROI_valid,1)
    Qu_ini = np.eye(n_ROI_valid)
    U_ini = np.random.randn(n_ROI_valid,q)*10
    #U_ini = U
    L_list_ini = L_list
    Sigma_ini = Sigma.copy()
    Result_one_step = coordinate_descent_opt( G,M,  Sigma_E, ROI_list, J_ini, dim,
                            U_ini, Qu_ini, L_list_ini , Sigma_ini, Mu_ini,
                            MaxIter = MaxIter, tol = tol, Flag_verbose = Flag_verbose)
    Result_two_step = coordinate_descent_opt( G,M,  Sigma_E, ROI_list, J_ini, dim,
                            MaxIter = 0, tol = 1E-4, Flag_verbose = False)
    print "truth"
    print Qu
    print "one-step"
    print Result_one_step['Qu']
    print "eigenvalues"
    print np.linalg.eigvalsh(Result_one_step['Qu'])
    print "two-step"
    print Result_two_step['Qu']
    
    if True:
        plt.close('all')
        
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(J,interpolation = "none", aspect = "auto")
        plt.title('True J')
        plt.colorbar()
        plt.subplot(1,3,2)
        #plt.imshow(Result_two_step['J'], interpolation = "none", aspect = "auto")
        plt.imshow(J_two_step, interpolation = "none", aspect = "auto")
        plt.title('two_step_J')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(Result_one_step['J'], interpolation = "none", aspect = "auto")
        plt.title('One_step_J')
        plt.colorbar()
    
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(J.ravel(),J_two_step.ravel(),'.')
        plt.xlabel('True J')
        plt.ylabel('two_step')
        plt.subplot(1,2,2)
        plt.plot(J.ravel(),Result_one_step['J'].ravel(),'.')
        plt.xlabel('True J')
        plt.ylabel('one_step')
   
    if True:
        plt.figure()
        plt.plot(U.T, 'r')
        plt.plot(Result_one_step['U'].T, 'g')
        plt.legend()
        
        plt.figure()
        plt.plot(U.ravel(), Result_one_step['U'].ravel(), '.')
        plt.title('plot of U')
        
        print "one_step_mu"
        print Result_one_step['Mu']
        print "true_mu"
        print Mu
        
        
    

 