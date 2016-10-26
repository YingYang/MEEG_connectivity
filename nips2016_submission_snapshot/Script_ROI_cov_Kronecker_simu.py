# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import mne
import sys
import scipy.io
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)
from ROI_cov_Kronecker import (get_MNE_inverse_sol,get_mle_kron_cov,sample_kron_cov,
                               get_neg_llh_kron,get_map_coor_descent_kron)                            
from ROI_cov import get_map_coor_descent, get_neg_llh
                               
import matplotlib.pyplot as plt
from mne.datasets import sample
import copy

if __name__ == "__main__":
    data_path = sample.data_path()
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
    fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
    G = fwd['sol']['data'][0:306,:]   
    
    ind0 = fwd['src'][0]['inuse']
    ind1 = fwd['src'][1]['inuse']
    # positions of dipoles
    rr = np.vstack([fwd['src'][0]['rr'][ind0==1,:], 
                             fwd['src'][1]['rr'][ind1==1,:]])
    rr = rr/np.max(np.sum(rr**2, axis = 1))                       
    nn = np.vstack([fwd['src'][0]['nn'][ind0 == 1,:],
                    fwd['src'][1]['nn'][ind1 == 1,:]])
                    
    # normalize G
    normalize_G_flag = True
    if normalize_G_flag:
        G /= np.sqrt(np.sum(G**2,axis = 0))
        
    n_channels, n_dipoles = G.shape
    src = fwd['src']
    m,n = n_dipoles, n_channels
    q = 300   
    T = 20
    #=========
    if True:
        label_names = ['mpfc-rh','mpfc-lh', 'phc-rh', 'phc-lh','lo-rh','lo-lh']
        ROI_dir = "/home/ying/Dropbox/MEG_source_loc_proj/Simulation_for_MLINI_2014/Simulation_Sample_Subjects/Hand_selected_ROIs"
        ROI_file_paths = [(ROI_dir + "/%s.label" % ln) for ln in label_names]
        n_ROI = len(ROI_file_paths)            
        labels = [mne.read_label(ROI_file_paths[ln]) for ln in range(n_ROI)]
        label_ind = list() 
        ROI0_ind = np.arange(0, n_dipoles, dtype = np.int)
        for i in range(len(labels)):
            _, sel = mne.source_space.label_src_vertno_sel(labels[i],src)
            label_ind.append(sel)
            ROI0_ind = np.setdiff1d(ROI0_ind, sel)
        label_ind.append(ROI0_ind)
        n_ROI = len(label_names)+1
        n_ROI_valid = n_ROI-1
    
    
    #========= load the parc labels
    if False:
        data_path = sample.data_path()
        subjects_dir = data_path + '/subjects'
        labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
        label_ind = list() 
        ROI0_ind = np.arange(0, n_dipoles, dtype = np.int)
        for i in range(len(labels)):
            _, sel = mne.source_space.label_src_vertno_sel(labels[i],src)
            label_ind.append(sel)
            ROI0_ind = np.setdiff1d(ROI0_ind, sel)
        label_ind.append(ROI0_ind)
        n_ROI = len(label_ind)
        n_ROI_valid = n_ROI-1
        #r = 10
        #tmp_vec =  np.random.randn(q, r)*np.random.randn(r)/r
        #Q = np.corrcoef(tmp_vec.T)  
        
    # check if number of dipoles is the same as the summed length of label_ind
    n_dipole_each_label = np.array([len(label_ind[l]) for l in range(len(label_ind)) ])    
    if n_dipole_each_label.sum()!= n_dipoles:
        raise ValueError("The label set not a parcellation of all sources")
        
    ROI_list = label_ind 
    L_list = list()
    for i in range(n_ROI_valid):
        tmp = np.ones( [len(ROI_list[i])])
        #tmp = tmp/np.linalg.norm(tmp)
        L_list.append( tmp) 
    
    a0,b0,c0 = 1.0, 2.0, 1.0 # a exp (-b ||x-y||^2)
    Q_L_list = list()
    for i in range(n_ROI_valid):
        tmp_n = len(ROI_list[i])
        tmp = np.zeros([tmp_n, tmp_n])
        for i0 in range(tmp_n):
            for i1 in range(tmp_n):
                tmp[i0,i1] = a0 * np.exp(-b0 * (
                np.sum((rr[i0,:]-rr[i1,:])**2)+ c0*(1- np.dot(nn[i0,:], nn[i1,:]))))
        #print np.linalg.cond(tmp)       
        Q_L_list.append(tmp)
        #Q_L_list.append(np.eye(tmp_n))
        
    inv_Q_L_list = copy.deepcopy(Q_L_list)
    for i in range(n_ROI_valid):
        inv_Q_L_list[i] = np.linalg.inv(Q_L_list[i])
    
    if False:
        L_list = list()
        for i in range(n_ROI_valid):
            tmp_n = len(ROI_list[i])
            tmp = np.random.multivariate_normal(np.zeros(tmp_n), Q_L_list[i])
            L_list.append(tmp)
        
    n_rep = 1
    mask = np.ones([n_ROI_valid,n_ROI_valid])
    mask = np.triu(mask,1)
    Lambda_seq = 10.0**np.arange(-2,3,1)
    n_Lambda = len(Lambda_seq)
    dist_to_true_corr = np.zeros([n_rep, n_Lambda+1])
    corr_with_true_corr =  np.zeros([n_rep, n_Lambda+1])
    corr_all_rep = np.zeros([n_rep, n_Lambda+2,n_ROI_valid, n_ROI_valid])
    for k in range(n_rep):
        if False:
            Q = np.eye(n_ROI_valid)*1.2
            #for l1 in range(n_ROI_valid//2):
            #    Q[2*l1,2*l1+1] = Q[2*l1+1,2*l1] = 0.4
            # randomly pick two numbers
            pick_times = n_ROI_valid**2/3
            for l1 in range(pick_times):
                tmp = np.random.choice(n_ROI_valid, 2, replace = False)
                Q[tmp[0], tmp[1]] = Q[tmp[1],tmp[0]] = (np.random.rand(1)*2-1)*0.6
            Q = Q*5
            
        if True:
            tmp = np.random.randn(n_ROI_valid, n_ROI_valid)
            r = np.random.gamma(shape=0.5, scale=1.0, size=n_ROI-1)
            Q = np.dot(tmp*r, (tmp*r).T)
            Q += np.eye(n_ROI_valid)
            diag1 = np.sqrt(np.diag(Q))
            denom1 = np.outer(diag1, diag1)
            Q = Q/denom1*5
            
        print (np.linalg.eigvalsh(Q)).min()
        # true corr
        diag1 = np.sqrt(np.diag(Q))
        denom1 = np.outer(diag1, diag1)
        corr_true = np.abs(Q/denom1)
        
        # create the true T
        a0,b0 = 1.0, 1E-1 # a exp (-b ||x-y||^2)
        # Gaussian process kernel for temporal smoothness
        Tcov = np.zeros([T,T])
        for i in range(T):
            for j in range(T):
                Tcov[i,j] = a0 * np.exp(-b0 * (i-j)**2)
        Tcov += 0.2*np.eye(T)
        print np.linalg.cond(Tcov)
        
        U = sample_kron_cov(Tcov, Q, n_sample = q)
        L = np.zeros([m, n_ROI_valid])
        for i in range(n_ROI_valid):
            L[ROI_list[i], i] = L_list[i]

        # test the marginal llh, whether the true parameter gives the best results
        L = np.zeros([m, n_ROI_valid])
        for i in range(n_ROI_valid):
            L[ROI_list[i], i] = L_list[i] 
        #Sigma_J_list = np.ones(n_ROI)*1
        Sigma_J_list = np.random.gamma(shape=2, scale=1.0, size=n_ROI)
        #Sigma_J_list/= np.mean(Sigma_J_list)
        Sigma_J = np.zeros(m)
        for i in range(n_ROI):
            Sigma_J[ROI_list[i]] = Sigma_J_list[i]    
            
        J = np.zeros([q,m,T])
        for i in range(n_ROI_valid):
            for j in range(len(ROI_list[i])):
                J[:,ROI_list[i][j],:] = L_list[i][j]*U[:,i,:] + (sample_kron_cov(Tcov, np.eye(1)*Sigma_J_list[i], n_sample = q))[:,0,:]
        if n_ROI_valid < n_ROI:
            i = -1
            for j in range(len(ROI_list[i])):
                J[:,ROI_list[i][j],:] = (sample_kron_cov(Tcov, np.eye(1)*Sigma_J_list[i], n_sample = q))[:,0,:]

        sigma = 5
        Sigma_E = np.eye(n)*sigma
        E = np.random.randn(q,n,T)*np.sqrt(sigma)
        M = ((J.transpose([0,2,1])).dot(G.T)).transpose([0,2,1]) + E
   

        #==== solving the problem ==========   
        # compute a two step result, this is only fair when Sigma_E is proportional to identity
        corr_two_step_all = np.zeros([n_Lambda, n_ROI_valid, n_ROI_valid])
        for i0 in range(n_Lambda):
            Lambda00 =  Lambda_seq[i0]
            # verify the MNE part is correct
            J_two_step = get_MNE_inverse_sol(M,G,Lambda00)
            U_two_step = np.zeros([q,n_ROI_valid,T])
            # extracting the ROI flipped mean, following mne.source_estimate.py extract_label_time_course, mode = 'mean_flip'
            # mne.label.py label_sign_flip
            # _, _, Vh = linalg.svd(ori, full_matrices=False)
            # flip = np.sign(np.dot(ori, Vh[:, 0] if len(vertno_sel) > 3 else Vh[0]))
            # I was using the row vector, why did they use the column vector? 
            # I have opened an issue for them
            for i in range(n_ROI_valid):
                J_tmp = J_two_step[:,ROI_list[i],:]
                tmp_nn = nn[ROI_list[i],:]
                tmpu, tmpd, tmpv = np.linalg.svd(tmp_nn)
                if False: # plot the eigen direction
                    from mpl_toolkits.mplot3d import Axes3D; fig = plt.figure(); ax = fig.add_subplot(111,projection = '3d')
                    ax.plot(tmp_nn[:,0], tmp_nn[:,1], tmp_nn[:,2], '.'); 
                    ax.plot(np.array([0,tmpv[0,0]]), np.array([0,tmpv[0,1]]), np.array([0,tmpv[0,2]]),'r-+'); plt.show()
                    ax.plot(np.array([0,tmpv[0,0]]), np.array([0,tmpv[1,0]]), np.array([0,tmpv[2,0]]),'g-+'); plt.show()
                tmp_sign = np.sign(np.dot(tmp_nn, tmpv[0]))
                U_two_step[:,i,:] = np.mean(J_tmp.transpose([0,2,1])*tmp_sign, axis = -1)
            Tcov_two_step, Qu_two_step = get_mle_kron_cov(U_two_step, tol = 1E-6, MaxIter = 100)
            diag2 = np.sqrt(np.diag(Qu_two_step))
            denom2 = np.outer(diag2, diag2)
            corr_two_step = np.abs(Qu_two_step/denom2)
            corr_two_step_all[i0] = corr_two_step
            
        # no idea how to do pairwise coorelation here?
            
        #=============== my method, Kronecker====
        nu = n_ROI_valid+1
        V = np.eye(n_ROI_valid)
        nu1 = T+1
        V1 = np.eye(T)
    
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
 
        prior_Q, prior_Sigma_J,prior_L, prior_Tcov = False, False, False, False
        alpha, beta = 1.0, 1.0
                   
        print "initial obj"
        obj0 = get_neg_llh_kron(Phi0, sigma_J_list0, L_list0, T00, # unknown parameters
                         ROI_list, G, M, q, 
                         nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                         prior_Q, prior_Sigma_J, prior_L, prior_Tcov) # prior flags
        print obj0                 
        print "optimial obj" 
        Phi = np.linalg.cholesky(Q) 
        sigma_J_list = np.sqrt(Sigma_J_list)
        T0 = np.linalg.cholesky(Tcov)                
        obj_star = get_neg_llh_kron(Phi, sigma_J_list, L_list, T0, # unknown parameters
                         ROI_list, G, M, q, 
                         nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                         prior_Q, prior_Sigma_J, prior_L, prior_Tcov) # prior flags 
        print obj_star
        Q_flag, Sigma_J_flag, L_flag, Tcov_flag  = True, True, False, True
        #L_list0 = copy.deepcopy(L_list)
        Qu_hat, Sigma_J_list_hat, L_list_hat, Tcov_hat, obj = get_map_coor_descent_kron(
                         Qu0, Sigma_J_list0, L_list0, Tcov0, # unknown parameters
                         ROI_list, G, M, q, 
                         nu, V, nu1, V1, inv_Q_L_list, alpha, beta, # prior params
                         prior_Q, prior_Sigma_J, prior_L, prior_Tcov, # prior flags
                         Q_flag = True, Sigma_J_flag = True, L_flag = True, Tcov_flag = True,
                         tau = 0.8, step_ini = 1.0, MaxIter = 10, tol = 1E-6, verbose = True, # optimization params
                         MaxIter0 = 20, tol0 = 1E-4, verbose0 = False)
       
        diag0 = np.sqrt(np.diag(Qu_hat))
        denom = np.outer(diag0, diag0)
        corr_hat = np.abs(Qu_hat/denom)
        
        
        MMT = M[:,:,0].dot(M[:,:,0].T)
        V_inv = V
        # static version
        Qu_hat_s, Sigma_J_list_hat_s, L_list_hat_s, obj = get_map_coor_descent(Qu0, Sigma_J_list0, L_list0,
                          ROI_list, G, MMT, q, Sigma_E,
                          nu, V_inv, inv_Q_L_list, alpha, beta, 
                          prior_Q, prior_Sigma_J, prior_L ,
                          Q_flag = Q_flag, Sigma_J_flag = Sigma_J_flag, L_flag = L_flag,
                          tau = 0.7, step_ini = 1.0, MaxIter = 15, tol = 1E-5,
                          eps = 1E-13, verbose = True, verbose0 = False, MaxIter0 = 10)  
        
        diag0 = np.sqrt(np.diag(Qu_hat))
        denom = np.outer(diag0, diag0)
        corr_hat = np.abs(Qu_hat/denom)
        

        if False:
            Q_list = [ Q, Qu_hat, Qu_two_step ]
            Q_legend = ['cov true', 'cov hat', 'cov two step']
            plt.figure()
            for l in range(len(Q_list)):
                plt.subplot(1,len(Q_list),l+1)
                plt.imshow(Q_list[l], vmin = None, vmax = None, interpolation ="none")
                plt.title(Q_legend[l])
                plt.colorbar()
                
            plt.figure(figsize = (10,8) )
            corr_names = ['true','one_step']
            for i0 in range(n_Lambda):
                corr_names.append('mne lambda = 10^%d' %np.log10(Lambda_seq[i0]))
            n_rows = 3
            n_cols = np.int(np.ceil(corr_all.shape[0]/np.float(n_rows)))
            for l in range(corr_all.shape[0]):
                _= plt.subplot(n_rows, n_cols ,l+1)
                _= plt.imshow(corr_all[l], vmin = 0, vmax =1, 
                              interpolation ="none", aspect = "auto")
                _=plt.colorbar()
                _=plt.title(corr_names[l])
            #_=plt.tight_layout()
            plt.savefig("/home/ying/Dropbox/tmp/ROI_cov_Kronecker_simu_example.pdf")
        
        corr_all = np.zeros([2+n_Lambda, n_ROI_valid, n_ROI_valid])
        corr_all[0] = corr_true
        corr_all[1] = corr_hat
        corr_all[2::] = corr_two_step_all
      
        corr_all_rep[k] = corr_all
        # first row is one_step, second row is two_step
        for l in range(corr_all.shape[0]-1):
            dist_to_true_corr[k,l] = np.linalg.norm(corr_all[l+1]- corr_all[0])
        tmp_triu0 = corr_all[0][mask>0]
        for l in range(corr_all.shape[0]-1):
            tmp_triu1 = corr_all[l+1][mask>0]
            corr_with_true_corr[k,l] = np.corrcoef(np.vstack([tmp_triu0, tmp_triu1]))[0,1]
        print dist_to_true_corr[k,:]
        print corr_with_true_corr[k,:]
    
    mat_name = "/home/ying/Dropbox/tmp/ROI_cov_Kronecker_simu.mat"
    mat_dict = dict(dist_to_true_corr = dist_to_true_corr, 
                    corr_with_true_corr = corr_with_true_corr,
                    corr_all_rep = corr_all_rep)
    scipy.io.savemat(mat_name, mat_dict)

    corr_with_true_corr_mne_flip = np.min(corr_with_true_corr[:,1:n_Lambda+1], axis = 1)
    method_names = ['one-step','mne-sign-flip',
                    'one-step\n -mne-sign-flip']
    dist_to_true_corr_all = np.zeros([n_rep,3])
    dist_to_true_corr_all[:,0] = dist_to_true_corr[:,0]
    dist_to_true_corr_all[:,1] = np.min(dist_to_true_corr[:,1:n_Lambda+1], axis = 1)
    dist_to_true_corr_all[:,2] = dist_to_true_corr_all[:,0]-dist_to_true_corr_all[:,1]
    
    corr_with_true_corr_all = np.zeros([n_rep,3])
    corr_with_true_corr_all[:,0] = corr_with_true_corr[:,0]
    corr_with_true_corr_all[:,1] = np.max(corr_with_true_corr[:,1:n_Lambda+1], axis = 1)
    corr_with_true_corr_all[:,2] = corr_with_true_corr_all[:,0] - corr_with_true_corr_all[:,1]
     
    n_method = len(method_names)
    data = [dist_to_true_corr_all, corr_with_true_corr_all]
    data_name = ['dist to true corr', 'corr with true core']
    
    xtick = np.arange(1,n_method+1)
    width = 0.8
    xticknames = method_names
    plt.figure(figsize = (8,6))
    for j0 in range(2):
        plt.subplot(2,1,j0+1)
        tmp_data = data[j0]
        plt.bar(xtick,np.mean(tmp_data, axis = 0), width = width)
        plt.errorbar(xtick+width/2, np.mean(tmp_data, axis = 0), 
                     np.std(tmp_data,axis = 0)/np.sqrt(n_rep), fmt = None)
        plt.xticks(xtick+width/2,xticknames)
        plt.xlim(0.3,n_method+1)
        plt.ylabel(data_name[j0])
        plt.grid()
    fig_name = "/home/ying/Dropbox/tmp/ROI_cov_simu.pdf"
    plt.savefig(fig_name)


    
    