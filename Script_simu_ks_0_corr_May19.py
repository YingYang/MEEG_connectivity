# -*- coding: utf-8 -*-
"""
    #fwd_path = mat_dict['fwd_path'][0]
    #print fwd_path
    #fwd = mne.read_forward_solution(fwd_path, force_fixed=True, surf_ori = True)
    #print np.linalg.norm(mat_dict['G'] - fwd['sol']['data'])
    u_array = mat_dict['u'].transpose([0,2,1])
    
    # debug, once the free-orentation fwd is saved and reloaded, G changes to weird values
    # Also, copying the file within python did not help. 
    #subj = "Subj1"
    #fwd_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/fwd/" \
    #                + "%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
    # keep fixed orientation                 
    #fwd = mne.read_forward_solution(fwd_fname, force_fixed=True, surf_ori=True)
    #mne.write_forward_solution(outpath+"-fwd.fif", fwd, overwrite = True) 
    # debug, why after saving, the same forward has totally different G
    #fwd1 = mne.read_forward_solution(outpath+"-fwd.fif", force_fixed=True, surf_ori=True )
    #print np.linalg.norm( mat_dict['G']- fwd1['sol']['data'])
"""
import numpy as np
import mne
import sys
import scipy.io
import copy

# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)         
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')              
import matplotlib.pyplot as plt
#from get_simu_data_ks import get_simu_data_ks
from get_simu_data_ks_ico_4 import get_simu_data_ks_ico_4
from get_estimate_baseline import get_estimate_baseline 
from get_estimate_ks import get_estimate_ks 
from ROI_Kalman_smoothing import get_param_given_u


#%% create simulations
#=============================================================================
simu_path = "/home/ying/dropbox_unsync/MEEG_source_roi_cov_simu_and_data/"
simu_id_start = 0
simu_id_end = 5
simu_id = range(simu_id_start, simu_id_end)
n_simu = len(simu_id)
T = 20
q = 200

old_simu = False

#======== use freesurfer anatomical labels
anat_ROI_names= ['pericalcarine-lh', 'pericalcarine-rh',
                 #'lateraloccipital-lh', 'lateraloccipital-rh',]
                'parahippocampal-lh', 'parahippocampal-rh',]
#                 'medialorbitofrontal-lh','medialorbitofrontal-rh']

subjects_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
subj = "Subj1"
labels0 = mne.read_labels_from_annot(subj, parc='aparc',
                                    subjects_dir=subjects_dir)
labels = list()
for i in range(len(anat_ROI_names)//2):
    tmp_label_list = list()
    for j in range(2):
        tmp_label = [label for label in labels0 if label.name in anat_ROI_names[2*i+j:2*i+j+1]]
        tmp_label_list.append(tmp_label[0])
    labels.append(tmp_label_list[0] + tmp_label_list[1])
p = len(labels)  
                    
#======== or use my hand-drawn labels, TBA  
alpha_list = np.array([2,5]); 
n_alpha = len(alpha_list);
# also define the gamma_distribution of sigma_i^2
flag_random_A = False
flag_time_smooth_source_noise = False
flag_space_smooth_source_noise = False
flag_nn_dot = True
flag_empi_true = flag_time_smooth_source_noise or flag_space_smooth_source_noise

lambda2_seq = np.exp(np.arange(-2,3))

#%% solving the simulations, using mne and ks
# define some function and call them is easier!!
def get_solution(outpath, lambda2_seq):                    
    #=======solution========================= 
    mat_dict = scipy.io.loadmat(outpath)
    ROI_list = list()
    n_ROI = len(mat_dict['ROI_list'][0])
    n_ROI_valid = n_ROI-1
    for i in range(n_ROI):
        ROI_list.append(mat_dict['ROI_list'][0,i][0])
    M = mat_dict['M']
    
    fwd_path = mat_dict['fwd_path'][0]
    noise_cov_path = mat_dict['noise_cov_path'][0]
    evoked_path = outpath+"-ave.fif.gz" 
    
    prior_A = dict(lambda0 = 0.0, lambda1 = 0.1)
    #prior_A = None
    prior_Q0, prior_Q, prior_sigma_J_list = None,None,None    
    MaxIter0, MaxIter = 100, 30
    tol0,tol = 1E-4, 2E-2
    verbose0, verbose = False, False
    L_flag = False
    whiten_flag = True
    depth=None
    flag_A_time_vary = True
    
    
    n_lambda2 = len(lambda2_seq)
    if False:
        for l0 in range(n_lambda2):
            tmp_lambda2 = lambda2_seq[l0]   
            for flag_sign_flip in [True, False]:  
                out_name_mne = outpath + "_mne_sol_lbdid%d_flip%d" %(l0, flag_sign_flip)
                get_estimate_baseline(M, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_mne, 
                         method = "MNE", lambda2 = tmp_lambda2, prior_Q0 = prior_Q0, 
                         prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                         prior_A = prior_A, depth = depth, MaxIter0 = MaxIter0, 
                         MaxIter = MaxIter, tol0 = tol0, tol = tol, verbose0 = verbose, 
                         verbose =verbose, flag_A_time_vary = flag_A_time_vary, 
                         flag_sign_flip = flag_sign_flip) 
        
    tmp_out_name_mne = outpath + "_mne_sol_lbdid%d_flip%d" %(n_lambda2//2, True) 
    result_mne = scipy.io.loadmat(tmp_out_name_mne)
    
    ini_Gamma0_list = [np.linalg.cholesky(result_mne['Q0_hat'])]
    ini_A_list = [result_mne['A_hat']] #
    ini_Gamma_list = [np.linalg.cholesky(result_mne['Q_hat'])]
    ini_sigma_J_list = [np.ones(len(ROI_list))*1E-9]

    # debugging
    #ini_Gamma0_list = [np.linalg.cholesky(mat_dict['Q0']), np.linalg.cholesky(result_mne['Q0_hat'])]
    #ini_A_list = [mat_dict['A'], result_mne['A_hat']] #
    #ini_Gamma_list = [np.linalg.cholesky(mat_dict['Q']), np.linalg.cholesky(result_mne['Q_hat'])]
    #ini_sigma_J_list = [np.sqrt(mat_dict['Sigma_J_list'][0]), np.ones(len(ROI_list))*1E-9]
    
    del(result_mne)
    
    out_name_ks = outpath + "_ks_sol"          
    get_estimate_ks(M, ROI_list, n_ROI_valid, fwd_path, evoked_path, noise_cov_path, out_name_ks, 
                     prior_Q0 = prior_Q0, prior_Q = prior_Q, prior_sigma_J_list = prior_sigma_J_list, 
                     prior_A = prior_A,
                     depth = depth, MaxIter0 = MaxIter0, MaxIter = MaxIter,
                     tol0 = tol0, tol = tol,
                     verbose0 = verbose0, verbose = verbose, verbose_coarse = False,
                     L_flag = L_flag, whiten_flag = whiten_flag, n_ini = -1, 
                     flag_A_time_vary = flag_A_time_vary, use_pool = False, 
                     MaxIter_coarse = 0, ini_Gamma0_list = ini_Gamma0_list,
                     ini_A_list = ini_A_list, ini_Gamma_list = ini_Gamma_list,
                     ini_sigma_J_list = ini_sigma_J_list)
    return 0

#=========== run the solutions
if True: 
    for i in range(n_alpha):
        for k in range(n_simu):
            alpha = alpha_list[i]
            outpath = simu_path + \
               "/%s_ROI_alpha%1.1f_simu%d_randA%d_smthns0" \
                %(p,alpha,simu_id[k], flag_random_A)
            get_solution(outpath, lambda2_seq)
                

#%%============== evaluate the results ===========
def get_eval(outpath, n_lambda2, flag_vis = False, flag_empi_true = False,
             flag_sign_flip_list = [True, False]):
    
    mat_dict = scipy.io.loadmat(outpath)
    Q = mat_dict['Q']
    #Q0 = mat_dict['Q0']
    A = mat_dict['A']
    
    ROI_list = list()
    n_ROI = len(mat_dict['ROI_list'][0])
    for i in range(n_ROI):
        ROI_list.append(mat_dict['ROI_list'][0,i][0])
    
    p = Q.shape[0]
    T = A.shape[0] 
    q = mat_dict['M'].shape[0]
    J = mat_dict['J']
    u_array_from_J = np.zeros([q,T+1,p])
    for r in range(q):
        for i in range(p):
            u_array_from_J[r,:,i] = np.mean(J[r, ROI_list[i],:], axis = 0)
            
    #Sigma_J_list = mat_dict['Sigma_J_list'][0]
    #L_list = list()
    #for i in range(n_ROI_valid):
    #    L_list.append(mat_dict['L_list'][0,i][0])
    u_array = mat_dict['u'].transpose([0,2,1])
    u_array_true = u_array_from_J.copy() if flag_empi_true else u_array.copy()
    # try to estimate A and Q from u_array  
    
    Gamma0_0 = np.eye(p)       
    A_0 = np.zeros([T,p,p])
    Gamma_0 = np.eye(p)
    # first run the non_prior version to get a global solution
    Gamma0_1, A_1, Gamma_1 = get_param_given_u(u_array_true, Gamma0_0, A_0, Gamma_0, 
       flag_A_time_vary = True,
       prior_Q0 = None, prior_A = None, prior_Q = None,
       MaxIter0 = 100, tol0 = 1E-4, verbose0 = False,
       MaxIter = 100, tol = 1E-4, verbose = False)
            
    if flag_empi_true:   
        A_true = A_1.copy()
        Q_true = Gamma_1.dot(Gamma_1.T)
    else:
        A_true = A.copy()
        Q_true = Q.copy()
        
    tmp_Q_corr = np.sqrt(np.diag(Q_true))
    Q_abs_corr = np.abs(Q_true/np.outer(tmp_Q_corr, tmp_Q_corr))
       
    sol_out_names = [outpath + "_ks_sol"]
    for l0 in range(n_lambda2):
        for flag_sign_flip in flag_sign_flip_list :  
            sol_out_names.append(outpath + "_mne_sol_lbdid%d_flip%d" %(l0, flag_sign_flip))
    
    n_sol = len(sol_out_names)
    A_error = np.zeros(n_sol)  # raw relative error
    A_abs_error = np.zeros(n_sol) # relative error of absolute value of A
    Q_error = np.zeros(n_sol)  # raw relative error 
    Q_abs_error = np.zeros(n_sol) # relative error of absolute value of correlation
    u_corr = np.zeros(n_sol)
    u_error = np.zeros(n_sol)

    for l0 in range(len(sol_out_names)):
        result = scipy.io.loadmat(sol_out_names[l0])
        A_hat, Q_hat, u_array_hat = result['A_hat'], result['Q_hat'], result['u_array_hat']
        
        A_error[l0] = np.sqrt(np.sum( (A_hat-A_true)**2 ))/np.sqrt( np.sum(A_true**2) )
        A_abs_error[l0] = np.sqrt(np.sum( (np.abs(A_hat)- np.abs(A_true) )**2 ))/np.sqrt(np.sum(np.abs(A_true)**2))
        Q_error[l0] = np.sqrt(np.sum( (Q_hat-Q_true)**2) )/np.sqrt(np.sum(Q_true**2))
        tmp = np.sqrt(np.diag(Q_hat))
        Q_abs_corr_hat = np.abs(Q_hat/np.outer(tmp, tmp))
        Q_abs_error[l0] = np.sqrt(np.sum( (Q_abs_corr_hat-Q_abs_corr)**2 ))/np.sqrt(np.sum(Q_abs_corr**2))
        
        #q = M.shape[0]
        corr_ts = np.zeros(p)
        # compute the correlation fo the time coureses
        #for r in range(q):
        for i in range(p):
            corr_ts[i] = np.abs(np.corrcoef(u_array_hat[:,:,i].ravel(), u_array_true[:,:,i].ravel())[0,1])
        u_corr[l0] = corr_ts.mean()
        u_error[l0] = np.sqrt(np.sum( (u_array_hat-u_array_true)**2) )/ np.sqrt(np.sum(u_array_true**2))
 
    # diff or ratio?
    # first one is alway ks
    eval_ks = dict(A_error = A_error[0], A_abs_error = A_abs_error[0],
                      Q_error = Q_error[0], Q_abs_error = Q_abs_error[0],
                     u_corr = u_corr[0], u_error = u_error[0])
    eval_mne = dict(A_error = A_error[1::].min(), A_abs_error = A_abs_error[1::].min(),
                      Q_error = Q_error[1::].min(), Q_abs_error = Q_abs_error[1::].min(),
                     u_corr = u_corr[1::].max(), u_error = u_error[1::].min()) 
    eval_diff = dict()
    for i in ['A_error', 'A_abs_error', 'Q_error', 'Q_abs_error', 'u_corr', 'u_error']:
        eval_diff[i] = eval_ks[i]-eval_mne[i] 
       
    # visualization
    if flag_vis:
        print "A error"
        print A_error
        print "u corr"
        print u_corr
        # load the results
        result_ks = scipy.io.loadmat(sol_out_names[0])
        # pick the one that has the highest correlation
        ind = np.argmax(u_corr[1::]) +1               
        result_mne = scipy.io.loadmat(sol_out_names[ind])
        # plot the for entries of A 
        names = ['truth','ss','mne']
        plt.figure(figsize =(6,3))
        count = 0
        prop = {'size': 10}
        for i0 in range(p):
            for i1 in range(p):
                plt.subplot(p,p, count+1)
                _ = plt.plot(A[:,i0,i1]);
                _ = plt.plot(result_ks['A_hat'][:,i0,i1])
                _ = plt.plot(result_mne['A_hat'][:,i0,i1])
                _ = plt.title('A[:,%d,%d]'%(i0+1,i1+1))
                if count == 0:
                    _ = plt.legend(names, loc = 8, ncol = 2, prop = prop )
                    _ = plt.xlabel('time index')                   
                    
                _ = plt.ylim(-1,1)
                count += 1
        plt.tight_layout(0.01)
        plt.savefig(outpath + "_A_plot.png", dpi = 400)
        
        # plot Q
        vmin, vmax = Q.min(), Q.max()          
        data_to_show = [Q, result_ks['Q_hat'], result_mne['Q_hat']]
        plt.figure(figsize = (6,2))
        for i in range(3):
            _=plt.subplot(1,3,i+1); 
            _=plt.imshow(data_to_show[i], interpolation = "none", aspect = "auto",
                         vmin = vmin, vmax = vmax);
            _=plt.colorbar()
            _=plt.title(names[i])
        plt.tight_layout()
        plt.savefig(outpath + "_Q_plot.pdf")
        
        # correlation with u_array
        plt.figure(figsize = (6,4))
        r0, i0 = 0,1
        plt.plot(u_array[r0,:,i0])
        plt.plot(result_ks['u_array_hat'][r0,:,i0])
        plt.plot(result_mne['u_array_hat'][r0,:,i0])
        plt.xlabel('time index')
        plt.ylabel('u in %dth ROI' %(i0+1))
        _ = plt.legend(names, loc = 8, ncol = 2, prop = prop )
        plt.tight_layout()
        plt.savefig(outpath + "_u_plot_one_trial.pdf")  
        
        plt.figure(figsize = (6,4))
        count = 0
        tmp_list = [result_ks, result_mne]
        for l0 in range(len(tmp_list)):
            u_array_hat = tmp_list[l0]['u_array_hat']
            for i in range(p):
                count += 1
                plt.subplot(2,p,count)
                plt.plot(u_array_hat[:,:,i].ravel(), u_array[:,:,i].ravel(), '.')    
                plt.title('%d_th ROI' % i)
        plt.tight_layout()
        plt.savefig(outpath + "_u_corr.pdf") 

    return eval_ks, eval_mne, eval_diff
    

#=========== run the evaluation
n_lambda2 = len(lambda2_seq)
if True: 
    crit_keys = ['A_error', 'A_abs_error', 'Q_error', 'Q_abs_error', 'u_corr', 'u_error']
    n_crit = len(crit_keys) # 5 different crierions
    # [ks, mne, diff]
    eval_result = np.zeros([n_crit, 3, n_alpha, n_simu])
    for i in range(n_alpha):
        for k in range(n_simu):
            alpha = alpha_list[i]
            if i == 1 and k ==0:
                flag_vis = True
            else:
                flag_vis = False
            
            # for old simulations
            if old_simu: 
                outpath = simu_path + "/%s_ROI_alpha%1.1f_simu%d_randA%d_smthns%d" \
                %(p,alpha,simu_id[k], flag_random_A, 0)
            else:
                outpath = simu_path + \
                   "/%s_ROI_alpha%1.1f_simu%d_randA%d_t%d_s%d_nn%d" \
                    %(p,alpha,simu_id[k], flag_random_A, 
                      flag_time_smooth_source_noise, 
                      flag_space_smooth_source_noise,
                      flag_nn_dot)
            eval_ks, eval_mne, eval_diff = get_eval(outpath, n_lambda2, flag_vis = flag_vis, 
                                           flag_sign_flip_list = [False], flag_empi_true = flag_empi_true)
            tmp = [eval_ks, eval_mne, eval_diff]
            for i1 in range(n_crit):
                for i2 in range(len(tmp)):
                    eval_result[i1,i2,i,k] = tmp[i2][crit_keys[i1]]
    # do bar plots for with std for each crit, different alpha,  [ks, mne, diff]                                   