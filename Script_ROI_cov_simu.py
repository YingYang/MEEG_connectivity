# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import mne
import sys
import scipy.io
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)
from One_and_two_step_regressions import two_step_regression
from ROI_cov import get_map_coor_descent, get_neg_llh
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
    if False:
        L_list = list()
        for i in range(n_ROI_valid):
            tmp = np.ones( [len(ROI_list[i])])
            #tmp = tmp/np.linalg.norm(tmp)
            L_list.append( tmp) 
    
    b0 = 1.5 # a exp (-b ||x-y||^2)
    Q_L_list = list()
    for i in range(n_ROI_valid):
        tmp_n = len(ROI_list[i])
        tmp = np.zeros([tmp_n, tmp_n])
        for i0 in range(tmp_n):
            for i1 in range(tmp_n):
                tmp[i0,i1] = np.dot(nn[i0,:], nn[i1,:])* np.exp(-b0 * (np.sum((rr[i0,:]-rr[i1,:])**2)))
        #print np.linalg.cond(tmp)       
        Q_L_list.append(tmp)
        #Q_L_list.append(np.eye(tmp_n))
        
    inv_Q_L_list = copy.deepcopy(Q_L_list)
    for i in range(n_ROI_valid):
        inv_Q_L_list[i] = np.linalg.inv(Q_L_list[i])
    
    if True:
        L_list = list()
        for i in range(n_ROI_valid):
            #tmp_n = len(ROI_list[i])
            #tmp = np.random.multivariate_normal(np.zeros(tmp_n), Q_L_list[i])
            # use sign alignment
            tmp_nn = nn[ROI_list[i],:]
            tmpu, tmpd, tmpv = np.linalg.svd(tmp_nn)
            tmp= np.sign(np.dot(tmp_nn, tmpv[0]))
            L_list.append(tmp)
            #L_list.append(tmpu[:,0])
            
        
        
    #n_rep = 10
    n_rep = 5
    mask = np.ones([n_ROI_valid,n_ROI_valid])
    mask = np.triu(mask,1)
    Lambda_seq = 10.0**np.arange(-2,3,1)
    n_Lambda = len(Lambda_seq)
    dist_to_true_corr = np.zeros([n_rep, 2*n_Lambda+1])
    corr_with_true_corr =  np.zeros([n_rep, 2*n_Lambda+1])
    corr_all_rep = np.zeros([n_rep, 2*n_Lambda+2,n_ROI_valid, n_ROI_valid])
    alpha = 1
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
            Q = Q* alpha
            
        if True:
            tmp = np.random.randn(n_ROI_valid, n_ROI_valid)
            r = np.random.gamma(shape=0.5, scale=1.0, size=n_ROI-1)
            Q = np.dot(tmp*r, (tmp*r).T)
            Q += np.eye(n_ROI_valid)
            diag1 = np.sqrt(np.diag(Q))
            denom1 = np.outer(diag1, diag1)
            Q = Q/denom1* alpha
            
        print (np.linalg.eigvalsh(Q)).min()
        
        Mu = np.zeros([n_ROI_valid,1])               
        U = np.random.multivariate_normal(Mu[:,0], Q, q).T
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
         
        JE0 = np.random.randn(m,q)
        JE = (JE0.T*np.sqrt(Sigma_J)).T 
        J = JE.copy()
        for i in range(n_ROI_valid):
            J[ROI_list[i],:] = JE[ROI_list[i],:] +np.outer(L_list[i], U[i,:])
               
        sigma = 5
        Sigma_E = np.eye(n)*sigma
        E = np.random.randn(n,q)*np.sqrt(sigma)
        M = np.dot(G, J) + E
   
        M_demean = (M.T - np.mean(M, axis = 1)).T
        MMT = M_demean.dot(M_demean.T)    
        GL = np.dot(G,L) 
        covM = MMT/q
        cov_ana = Sigma_E + np.dot(G, np.diag(Sigma_J)).dot(G.T) + np.dot(GL, Q).dot(GL.T)
        
        # true corr
        diag1 = np.sqrt(np.diag(Q))
        denom1 = np.outer(diag1, diag1)
        corr_true = np.abs(Q/denom1)
        #============= Debug the simulation procedure =======================: 
        if False:
            plt.figure()
            plt.subplot(1,3,1);plt.imshow(covM, interpolation = "none");plt.colorbar()
            plt.subplot(1,3,2);plt.imshow(cov_ana, interpolation = "none");plt.colorbar()
            plt.subplot(1,3,3);plt.imshow(cov_ana-covM, interpolation = "none");plt.colorbar()
            print np.linalg.norm(covM-cov_ana)/np.linalg.norm(covM)
            print np.linalg.norm(covM-cov_ana)
            plt.figure();plt.plot(covM.ravel(), cov_ana.ravel(), '.');
        #===================end of debug ========================
        #==== solving the problem ==========   
        # compute a two step result, this is only fair when Sigma_E is proportional to identity
        # I should allow gamma to change
        X = np.ones([q,1])
        corr_two_step_all = np.zeros([n_Lambda, n_ROI_valid, n_ROI_valid])
        corr_two_step_pair_ave_all = np.zeros([n_Lambda, n_ROI_valid, n_ROI_valid])
        for i0 in range(n_Lambda):
            Lambda00 =  Lambda_seq[i0]
            # verify the MNE part is correct
            J_two_step, Beta_two_step= two_step_regression(M, G, X.T, Lambda00, 0.0)
            #J_two_step1 = (np.linalg.inv(G.T.dot(G) + np.diag(Lambda00*np.ones(m)))).dot(G.T.dot(M))            
            #print np.linalg.norm(J_two_step1-J_two_step) /  np.linalg.norm(J_two_step1)        
            # estimate the hidden U for each ROI, and then compute teh two-step cov
            U_two_step = np.zeros([n_ROI_valid, q])
            for i in range(n_ROI_valid):
                J_tmp = J_two_step[ROI_list[i],:]
                #L_tmp = L_list[i]
                #U_two_step[i] = 1.0/np.sum(L_tmp**2)* np.dot(J_tmp.T, L_tmp)
                #==== use PCA to get it
                #J_tmp = (J_tmp.T- np.mean(J_tmp, axis = 1)).T
                #tmpu, tmpd, tmpv = np.linalg.svd(J_tmp, full_matrices = 0)
                #U_two_step[i] = tmpv[0,:]
                #==== use the align_z method by MNE, align with the principle normal direction
                tmp_nn = nn[ROI_list[i],:]
                tmpu, tmpd, tmpv = np.linalg.svd(tmp_nn)
                tmp_sign = np.sign(np.dot(tmp_nn, tmpv[0]))
                U_two_step[i] = np.mean(J_tmp.T * tmp_sign, axis = 1)
            Qu_two_step = np.cov(U_two_step)
            diag2 = np.sqrt(np.diag(Qu_two_step))
            denom2 = np.outer(diag2, diag2)
            corr_two_step = np.abs(Qu_two_step/denom2)
            corr_two_step_all[i0] = corr_two_step
            
            # alternative: compute the averaged pairwise correlation of sources between two ROIs
            corr_two_step_pair_ave = np.eye(n_ROI_valid)
            for l1 in range(n_ROI_valid):
                for l2 in range(l1+1, n_ROI_valid):
                    J_tmp1 = J_two_step[ROI_list[l1],:]
                    J_tmp2 = J_two_step[ROI_list[l2],:]
                    tmp_corr = np.corrcoef(np.vstack([J_tmp1, J_tmp2]))
                    tmp_corr_valid = tmp_corr[0:J_tmp1.shape[0], J_tmp1.shape[0]::]
                    corr_two_step_pair_ave[l1,l2] = np.mean(np.abs(tmp_corr_valid))
                    corr_two_step_pair_ave[l2,l1] = corr_two_step_pair_ave[l1,l2]
            corr_two_step_pair_ave_all[i0] = corr_two_step_pair_ave
            
        #=============== my method
        # nu > p-1 is required for the inverse wishart to work
        nu = n_ROI_valid +1
        V_inv = np.eye(n_ROI_valid)*1E-4
        prior_Q, prior_Sigma_J, prior_L = False, False, True
        alpha, beta = 1.0, 1.0
        inv_Q_L_list = copy.deepcopy(Q_L_list)
        for i in range(n_ROI_valid):
            inv_Q_L_list[i] = np.linalg.inv(Q_L_list[i]) 
        
        eps = 1E-13
        Qu0 = np.eye(n_ROI_valid)    
        sigma_J_list0 = np.ones(n_ROI)
        Sigma_J_list0 = sigma_J_list0**2
         
        L_list0 = copy.deepcopy(L_list)
        for i in range(n_ROI_valid):
            #L_list0[i] = np.random.randn(L_list0[i].size)
            L_list0[i] = np.ones(L_list0[i].size)
            
        print "initial obj"
        Phi0 = np.linalg.cholesky(Qu0)       
        sigma_J_list0 = np.sqrt(Sigma_J_list0)
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
        
        # maybe not update L at all, updating L causes problems
        # How to constrain L
        #L_list0 = copy.deepcopy(L_list)
        #Sigma_J_list0 = copy.deepcopy(Sigma_J_list)
        Q_flag, Sigma_J_flag, L_flag = True, True, True
        Qu_hat, Sigma_J_list_hat, L_list_hat, obj = get_map_coor_descent(Qu0, Sigma_J_list0, L_list0,
                          ROI_list, G, MMT, q, Sigma_E,
                          nu, V_inv, inv_Q_L_list, alpha, beta, 
                          prior_Q, prior_Sigma_J, prior_L ,
                          Q_flag = Q_flag, Sigma_J_flag = Sigma_J_flag, L_flag = L_flag,
                          tau = 0.7, step_ini = 1.0, MaxIter = 20, tol = 1E-5,
                          eps = 1E-13, verbose = True, verbose0 = False, MaxIter0 = 10)                       
        #if False:
        #    Qu0 = Qu_hat
        #    Sigma_J_list0 = Sigma_J_list_hat
        #    L_list0 = L_list_hat                  
        diag0 = np.sqrt(np.diag(Qu_hat))
        denom = np.outer(diag0, diag0)
        corr_hat = np.abs(Qu_hat/denom)
        
        # try miss one of the ROIs
        if False:
            # randomly remove two ROIs to model
            n_ROI_subset_valid = 4
            ind_roi_subset = np.sort(np.random.choice(np.arange(n_ROI_valid), 
                            n_ROI_subset_valid, replace = False))
            Q_subset = Q[ind_roi_subset,:]
            Q_subset = Q_subset[:,ind_roi_subset]
            diag_subset = np.sqrt(np.diag(Q_subset))
            corr_subset = np.abs(Q_subset/np.outer(diag_subset, diag_subset))
            
            Qu_hat_subset = Qu_hat[ind_roi_subset,:]
            Qu_hat_subset = Qu_hat_subset[:,ind_roi_subset]
            diag_subset = np.sqrt(np.diag(Qu_hat_subset))
            corr_subset = np.abs(Q_subset/np.outer(diag_subset, diag_subset))
            
            
            ROI_list1 = [ROI_list[i] for i in range(n_ROI_valid) if i in ind_roi_subset]
            ROI0_ind = np.arange(0, n_dipoles, dtype = np.int)
            for i in range(n_ROI_subset_valid):
                ROI0_ind = np.setdiff1d(ROI0_ind, ROI_list1[i])
            ROI_list1.append(ROI0_ind) 
            Qu0_subset = np.eye(n_ROI_subset_valid)
            Sigma_J_list0_subset = np.ones(n_ROI_subset_valid+1)
            L_list0_subset = [L_list0[i] for i in range(n_ROI_valid) if i in ind_roi_subset]
            V_inv_subset = V_inv[ind_roi_subset]
            V_inv_subset = V_inv_subset[:,ind_roi_subset]
            inv_Q_L_list_subset = [inv_Q_L_list[i] for i in range(n_ROI_valid) if i in ind_roi_subset]
            Qu_hat1, Sigma_J_list_hat1, L_list_hat1, obj1 = get_map_coor_descent(
                             Qu0_subset, Sigma_J_list0_subset, L_list0_subset,
                              ROI_list1, G, MMT, q, Sigma_E,
                              nu, V_inv_subset, inv_Q_L_list_subset, alpha, beta, 
                              prior_Q, prior_Sigma_J, prior_L ,
                              Q_flag = Q_flag, Sigma_J_flag = Sigma_J_flag, L_flag = L_flag,
                              tau = 0.7, step_ini = 1.0, MaxIter = 20, tol = 1E-6,
                              eps = 1E-13, verbose = True, verbose0 = False, MaxIter0 = 10)                       
            diag1= np.sqrt(np.diag(Qu_hat1))
            denom = np.outer(diag1, diag1)
            corr_hat1 = np.abs(Qu_hat1/denom)
        
            
            plt.figure()
            tmp_corr_data = [corr_true, corr_subset,  corr_hat1]
            tmp_corr_data_names = ['true','true subset','partial']
            for i0 in range(3):
                plt.subplot(1,3,i0+1)
                plt.imshow(tmp_corr_data[i0], vmin = 0, vmax = 1, interpolation = "None"); plt.colorbar()
                plt.title(tmp_corr_data_names[i0])
            plt.savefig("/home/ying/Dropbox/tmp/ROI_cov_simu_example_2ROI_missing.pdf")
        
        
        
        # plot the L_list
        if False:
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
            plt.plot(Sigma_J_list, Sigma_J_list_hat, '.');
            plt.xlabel('True variance of ROIs'); plt.ylabel('Estimated variance of ROIs');
  

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
            for i0 in range(n_Lambda):
                corr_names.append('mne pairwise lambda = 10^%d' %np.log10(Lambda_seq[i0]))
            n_rows = 3
            n_cols = np.int(np.ceil(corr_all.shape[0]/n_rows))
            for l in range(corr_all.shape[0]):
                _= plt.subplot(n_rows, n_cols ,l+1)
                _= plt.imshow(corr_all[l], vmin = 0, vmax =1, 
                              interpolation ="none", aspect = "auto")
                _=plt.colorbar()
                _=plt.title(corr_names[l])
            #_=plt.tight_layout()
            #plt.savefig("/home/ying/Dropbox/tmp/ROI_cov_simu_example.pdf")

                  
        corr_all = np.zeros([2+2*n_Lambda, n_ROI_valid, n_ROI_valid])
        corr_all[0] = corr_true
        corr_all[1] = corr_hat
        corr_all[2:2+n_Lambda] = corr_two_step_all
        corr_all[2+n_Lambda::] = corr_two_step_pair_ave_all
        
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
    
    mat_name = "/home/ying/Dropbox/tmp/ROI_cov_simu.mat"
    mat_dict = dict(dist_to_true_corr = dist_to_true_corr, 
                    corr_with_true_corr = corr_with_true_corr,
                    corr_all_rep = corr_all_rep)
    scipy.io.savemat(mat_name, mat_dict)
    

    corr_with_true_corr_mne_flip = np.min(corr_with_true_corr[:,1:n_Lambda+1], axis = 1)
    corr_with_true_corr_mne_pair = np.min(corr_with_true_corr[:,n_Lambda+1::], axis = 1)
    
    method_names = ['one-step','mne-sign-flip','mne-pairwise',
                    'one-step\n -mne-sign-flip',
                    'one-step\n -mne-pairwise']
    dist_to_true_corr_all = np.zeros([n_rep,5])
    dist_to_true_corr_all[:,0] = dist_to_true_corr[:,0]
    dist_to_true_corr_all[:,1] = np.min(dist_to_true_corr[:,1:n_Lambda+1], axis = 1)
    dist_to_true_corr_all[:,2] = np.min(dist_to_true_corr[:,n_Lambda+1::], axis = 1)
    dist_to_true_corr_all[:,3] = dist_to_true_corr_all[:,0]-dist_to_true_corr_all[:,1]
    dist_to_true_corr_all[:,4] = dist_to_true_corr_all[:,0]-dist_to_true_corr_all[:,2]
    
    corr_with_true_corr_all = np.zeros([n_rep,5])
    corr_with_true_corr_all[:,0] = corr_with_true_corr[:,0]
    corr_with_true_corr_all[:,1] = np.max(corr_with_true_corr[:,1:n_Lambda+1], axis = 1)
    corr_with_true_corr_all[:,2] = np.max(corr_with_true_corr[:,n_Lambda+1::], axis = 1)
    corr_with_true_corr_all[:,3] = corr_with_true_corr_all[:,0] - corr_with_true_corr_all[:,1]
    corr_with_true_corr_all[:,4] = corr_with_true_corr_all[:,0] - corr_with_true_corr_all[:,2]
    
    n_method = len(method_names)
    data = [dist_to_true_corr_all, corr_with_true_corr_all]
    data_name = ['(diff) dist to true corr', '(diff) corr with true corr']
    
    xtick = np.arange(1,n_method+1)
    width = 0.8
    xticknames = method_names
    plt.figure(figsize = (10,6))
    for j0 in range(2):
        plt.subplot(2,1,j0+1)
        tmp_data = data[j0]
        plt.bar(xtick,np.mean(tmp_data, axis = 0), width = width)
        plt.errorbar(xtick+width/2, np.mean(tmp_data, axis = 0), 
                     np.std(tmp_data,axis = 0)/np.sqrt(n_rep), fmt = None, lw = 2)
        plt.xticks(xtick+width/2,xticknames)
        plt.xlim(0.3,n_method+1)
        plt.ylabel(data_name[j0])
        plt.grid()
    fig_name = "/home/ying/Dropbox/tmp/ROI_cov_simu_L%d_alpha%d.pdf" %(L_flag, alpha)
    plt.savefig(fig_name)


    
    