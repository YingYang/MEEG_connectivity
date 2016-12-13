# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import scipy.stats
import copy

# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)   
import matplotlib
#matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#matplotlib.use('Agg')                    
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'plasma'
#from get_simu_data_ks import get_simu_data_ks
from ROI_Kalman_smoothing import get_cov_u

path = [ "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Connectivity_Analysis/",
         "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/",
         "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"]

for path0 in path:      
    sys.path.insert(0, path0)
    
from conn_utility import (get_tf_PLV, get_corr_tf)
from ar_utility import (least_square_lagged_regression, get_tilde_A_from_At, get_dynamic_linear_coef)   
from ROI_Kalman_smoothing import (get_param_given_u)
from Stat_Utility import Fisher_method


#=============================================================================================
if False:
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
    isMEG = True
    MEGorEEG = ['EEG','MEG']
    
    
    if isMEG:
        #subj_list = [1,2,3,5,6,7,9]
        #btstrp_seq = [20,20,20,20,20,8,20]
        subj_list = range(1,10) + range(11,13) + range(14, 19)
        btstrp_seq = (np.ones(len(subj_list))*29).astype(np.int)
    else:
        pass
        
    
    n_subj = len(subj_list)
    
    pairs = [[0,1],[1,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,5]]
    #pairs = [[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]]
    m1,m2 = 4,2
    n_pairs = len(pairs)
        
    #mne_method = 'MNE'
    #ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g'] #, 'LO_c_g']
    #label_names = ['EVC','PPA']
        
    ROI_bihemi_names = ['pericalcarine', 'PPA_c_g', 'TOS_c_g', 'RSC_c_g', 'LOC_c_g', 'medialorbitofrontal'] 
    ROI_names = ['EVC','PPA','TOS','RSC','LOC','mOFC'] 
    mne_method = 'dSPM'    
        
    p = len(ROI_bihemi_names)
    #method_string = ['ks','dspm']
    
    #method_string = ['ks','dSPM']
    method_string = ['dSPM']
    flag_cov_from_u = False
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/roi_ks/" 
    
    fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    #fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
    
    T0 = 95
    #n_method = len(method_string)
    n_method = 1
    A_est_all = np.zeros([n_method, n_subj, T0, p,p])
    A_se_all  = np.zeros([n_method, n_subj, T0, p,p])
    
    tildaA_all = np.zeros([n_method, n_subj, T0, T0])
    tildaA_Z_all   = np.zeros([n_method, n_subj, T0, T0])
    
    
    
    
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        tmp_bootstrap_seq = range(btstrp_seq[i])
        n_bt = len(tmp_bootstrap_seq)
        
        MEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
        EEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
    
        if isMEG:
            fwd_path = MEG_data_dir + "fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            epochs_path = MEG_data_dir + "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica-epo.fif.gz" %(subj,subj)
            ave_mat_path =  MEG_data_dir + "epoch_raw_data/%s/%s_%s.mat" %(subj,  subj, fname_suffix)
        else:
            fwd_path = EEG_data_dir + "%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)      
            epochs_path = EEG_data_dir + "epoch_raw_data/%s_EEG/%s_EEG_filter_1_110Hz_notch_ica_reref-epo.fif.gz" %(subj,subj)
            ave_mat_path =  EEG_data_dir + "epoch_raw_data/%s_EEG/%s_EEG_%s.mat" %(subj,  subj, fname_suffix)    
        
        # load the time indices
        mat_dict = scipy.io.loadmat(ave_mat_path)
        times = mat_dict['times'][0]
        del(mat_dict)
        offset = 0.04 if isMEG else 0.00
        time_ind = np.all(np.vstack([times >= -0.05, times <= 0.90 ]), axis = 0)
        times_in_ms = (times[time_ind]-offset)*1000.0
        print len(times_in_ms)    
        T = len(times_in_ms)-1
        
     
        A_cell = np.zeros([n_method,n_bt,T,p,p])
        lagged_corr = np.zeros([n_method,n_bt,n_pairs,T+1,T+1])
        lagged_inv = np.zeros([n_method, n_bt, n_pairs,T+1,T+1])
        std_u = np.zeros([n_method, n_bt,T+1,p])
        tilda_A_entry = np.zeros([n_method,n_bt,n_pairs, T,T])
        for l in range(n_method):
            for i00 in range(n_bt):
                bootstrap_id = tmp_bootstrap_seq[i00]
                #out_name = outdir + "%s_%s_%s_%s_%s_%s_sol_bootstrp%d.mat" % (subj, 
                #        "EVC_PPA", "ico41", fname_suffix, MEGorEEG[isMEG], method_string[l],bootstrap_id)
                out_name = outdir + "%s_%s_%s_%s_%s_%s_sol_bootstrp%d.mat" % (subj, 
                        "6ROI", "ico40", fname_suffix, MEGorEEG[isMEG], method_string[l],bootstrap_id)
                result = scipy.io.loadmat(out_name)
                A_cell[l,i00, :,:,:] = result['A_hat']
                u_array_hat = result['u_array_hat']
                std_u[l,i00] = np.std(u_array_hat, axis = 0)
                
                if flag_cov_from_u:       
                    cov_u = np.cov(u_array_hat.reshape([u_array_hat.shape[0], -1]).T)    
                else:
                    cov_u = get_cov_u(result['Q0_hat'], result['A_hat'], result['Q_hat'], T, 
                                      flag_A_time_vary = True) # pT x pT, p first 
                paired_lags = np.zeros([n_pairs, (T+1), (T+1)])
                for ii in range(n_pairs):
                    for t1 in range(T+1):
                        for t2 in range(T+1):
                            paired_lags[ii,t1,t2] = cov_u[t1*p+pairs[ii][0], t2*p+pairs[ii][1]]\
                            /np.sqrt(cov_u[t1*p+pairs[ii][0], t1*p+pairs[ii][0]])/ np.sqrt(cov_u[t2*p+pairs[ii][1], t2*p+pairs[ii][1]])
                    lagged_corr[l, i00] = paired_lags            
                inv_cov_u = np.linalg.inv(cov_u)
                #plt.figure(); plt.imshow(inv_cov_u, interpolation = "none"); plt.colorbar();
                paired_lags_inv = np.zeros([n_pairs, (T+1), (T+1)])
                for ii in range(n_pairs):
                    for t1 in range(T+1):
                        for t2 in range(T+1):
                            paired_lags_inv[ii,t1,t2] = inv_cov_u[t1*p+pairs[ii][0], t2*p+pairs[ii][1]]
                    lagged_inv[l, i00] = paired_lags_inv
    
                # compute u_i1 = \tilda A u_i2
                print "computing lagged A entries"
                tilde_A = np.zeros([T,T],dtype = np.object)  
                A = result['A_hat']
                for i0 in range(T):
                    tilde_A[i0,i0] = A[i0].copy()
                for i0 in range(T):
                    for j0 in range(i0+1,T):
                        tmp = np.eye(p)
                        for l0 in range(j0,i0-1,-1):
                            tmp = (np.dot(tmp, A[l0])).copy()
                        tilde_A[i0,j0] = tmp
                for ii in range(n_pairs):
                    for t1 in range(T):
                        for t2 in range(T):
                            if t1<= t2:
                                tilda_A_entry[l,i00, ii,t1,t2] = tilde_A[t1,t2][pairs[ii][1], pairs[ii][0]]
                            else:
                                # region 2 -> region 1 lower diangonal feedback
                                tilda_A_entry[l,i00, ii,t1,t2] = tilde_A[t2,t1][pairs[ii][0], pairs[ii][1]]
                del(result)
    
        # plot the all parts in A, with bootstrap CI
        print "plotting A"
        A_est = A_cell[:,0]
        A_se = np.std(A_cell[:,1::], axis = 1)
        
        # save into the all matrix
        A_est_all[:,i,:,:,:] = A_est
        A_se_all[:,i,:,:,:] = A_se
        
        var_u_est = std_u[:,0]**2
        var_u_se = np.std(std_u[:,1::]**2, axis = 1)
        
        print "std_u.shape"
        print std_u.shape
        print var_u_est.shape
        print var_u_se.shape
        
        alpha0 = scipy.stats.norm.ppf(1- 0.05/2)
        for l in range(n_method):
            plt.figure()
            count = 0
            ymin = None
            ymax = None
            for l1 in range(p):    
                # first dim is method
                _= plt.errorbar(times_in_ms, var_u_est[l,:,l1], alpha0*var_u_se[l,:,l1])
                _= plt.xlabel('time (ms)')
                #_= plt.plot(times_in_ms[1::], np.zeros(T), 'k')
                _= plt.ylim(ymin, ymax)
            #plt.legend(label_names)
            fig_name = fig_outdir + "%s_%s_var_u.pdf" %(subj, method_string[l])
            plt.tight_layout()
            #plt.savefig(fig_name)
            #plt.close('all')
            
        
        print A_est.shape
        print A_se.shape
        print A_est[0,2::2,0,0]
        print A_se[0,2::2,0,0]
    
        ymin = None
        ymax = None
        alpha = scipy.stats.norm.ppf(1- 0.05/2)
        for l in range(n_method):
            plt.figure()
            count = 0
            for l1 in range(p):
                for l2 in range(p):
                    _= plt.subplot(p,p, count+1);
                    # first dim is method
                    _= plt.errorbar(times_in_ms[1::], A_est[l,:,l1,l2], alpha*A_se[l,:,l1,l2])
                    _= plt.xlabel('time (ms)')
                    _= plt.title('A[:,%d,%d]'% (l1,l2))
                    _= plt.plot(times_in_ms[1::], np.zeros(T), 'k')
                    _= plt.ylim(ymin, ymax)
                    count += 1
            fig_name = fig_outdir + "%s_%s_A.pdf" %(subj, method_string[l])
            plt.tight_layout()
            #plt.savefig(fig_name)
            plt.close('all')
            
        # just plot A_est alone
        for l in range(n_method):
            plt.figure()
            count = 0
            ymin = -1.5
            ymax = 1.5
            print ymin, ymax
            ymin, ymax = A_est[l].min, A_est[l].max
            for l1 in range(p):
                for l2 in range(p):
                    _= plt.subplot(p,p, count+1);
                    # first dim is method
                    _= plt.plot(times_in_ms[1::], A_est[l,:,l1,l2])
                    _= plt.xlabel('time (ms)')
                    _= plt.title('A[:,%d,%d]'% (l1,l2))
                    _= plt.plot(times_in_ms[1::], np.zeros(T), 'k')
                    #_= plt.ylim(ymin, ymax)
                    count += 1
            fig_name = fig_outdir + "%s_%s_A_est.pdf" %(subj, method_string[l])
            plt.tight_layout()
            plt.savefig(fig_name)
            plt.close('all')
        
            
        print "plotting lagged cov"
        # plot the lagged correlation:
        """
        lag_list = [lagged_corr, lagged_inv]
        lag_names = ['corr','inv']
        vmin, vmax = 0,10
        for flag_thresh in [True, False]:
            for j in range(len(lag_list)):
                tmp = lag_list[j]
                est = tmp[:,0]
                se = np.std(tmp[:,1::], axis = 1)
                tmp_T = est/se 
                # bonferroni of T
                T_thresh = np.abs(scipy.stats.norm.ppf(0.05/2.0/np.float(T+1)**2))       
                tmp_T[np.isnan(tmp_T)] = 0
                tmp_T[np.isinf(tmp_T)] = 0
                if flag_thresh:
                    tmp_T[np.abs(tmp_T)<= T_thresh] = 0
                for l in range(2):
                    plt.figure(figsize = (13,3))
                    for i in range(n_pairs):
                        _= plt.subplot(m1,m2,i+1)
                        _= plt.imshow(np.abs(tmp_T[l,i]), interpolation = "none", aspect = "auto", 
                                   extent = [times_in_ms[0], times_in_ms[-1], 
                                times_in_ms[0], times_in_ms[-1]],origin= "lower", vmin = vmin, vmax = vmax)
                        _=plt.ylabel(label_names[pairs[i][0]] + " time (ms)");
                        _=plt.xlabel(label_names[pairs[i][1]] + " time (ms)")
                        _=plt.colorbar()
                        #_= plt.xlabel('ms')
                        #_= plt.ylabel('ms')
                    plt.tight_layout()
                    #fig_name = fig_outdir + "%s_%s_%s_thresh%d.pdf" %(subj, method_string[l], lag_names[j], flag_thresh)
                    #plt.savefig(fig_name)
                    #plt.close()
        """           
        
        #T_thresh = np.abs(scipy.stats.norm.ppf(0.1/2.0/np.float(T+1)**2)) 
        # only do the relevant pairs
           
        
        alpha = 0.05
        pair_id = 0    
        vmin, vmax = 0, 10
        figsize = (4.6,4)
        fdr_flag =False
        for flag_thresh in [True, False]:
            for l in range(n_method):
                print "saving tilde A"          
                tmp = tilda_A_entry
                est = tmp[l,0,pair_id]
                se = np.std(tmp[l,1::, pair_id,:,:], axis = 0)
                tmp_T = est/se
                tmp_T[np.isnan(tmp_T)] = 0
                tmp_T[np.isinf(tmp_T)] = 0
                
                tildaA_Z_all[l,i] = tmp_T
                # bonferroni of T 
                if flag_thresh:
                    # use FDR
                    #tmp_T[np.abs(tmp_T)<= T_thresh] = 0
                    if fdr_flag:
                        tmp_p_val = 2.0*(1-scipy.stats.norm.cdf(np.abs(tmp_T)))
                        reject, _ = mne.stats.fdr_correction(tmp_p_val.ravel(), alpha = alpha)
                        tmp_T1 = np.abs(tmp_T)
                        tmp_T1[np.reshape(reject, [T,T]) == 0] = 0.0
                    else:
                        T_thresh = np.abs(scipy.stats.norm.ppf(alpha/2.0/tmp_T.size))
                        tmp_T1 = np.abs(tmp_T)
                        tmp_T1[tmp_T1<=T_thresh] = 0
                else:
                    tmp_T1 = np.abs(tmp_T)
                    
                plt.figure(figsize = figsize)
                _= plt.imshow(tmp_T1, interpolation = "none", aspect = "auto", 
                           extent = [times_in_ms[0], times_in_ms[-1], 
                        times_in_ms[0], times_in_ms[-1]],origin= "lower", vmin = vmin, vmax = vmax)
                _=plt.ylabel(ROI_names[pairs[pair_id][0]]+ " time (ms)"); 
                _=plt.xlabel(ROI_names[pairs[pair_id][1]]+ " time (ms)")
                _=plt.colorbar()
                plt.tight_layout()
                fig_name = fig_outdir + "%s_%s_%s_thresh%d_fdr%d_%s.pdf" %(subj, method_string[l], 'tildeA', flag_thresh, fdr_flag, fname_suffix)
                plt.savefig(fig_name)
                plt.close()
                
        tildaA_all[:,i] = tilda_A_entry[:,0,pair_id,:,:] 
        
        # just show the estimated tildeA alone
        """
        for l in range(2):
            plt.figure(figsize = (13,3))
            for i in range(n_pairs):
                _= plt.subplot(m1,m2,i+1)
                _= plt.imshow(tilda_A_entry[l,0,i], interpolation = "none", aspect = "auto", 
                           extent = [times_in_ms[0], times_in_ms[-1], 
                        times_in_ms[0], times_in_ms[-1]],origin= "lower", vmin = None, vmax = None)
                _=plt.ylabel(label_names[pairs[i][0]]); plt.xlabel(label_names[pairs[i][1]])
                _=plt.colorbar()
                #_= plt.xlabel('ms')
                #_= plt.ylabel('ms')
            plt.tight_layout()
            fig_name = fig_outdir + "%s_%s_%s_est.pdf" %(subj, method_string[l], 'tildeA')
            plt.savefig(fig_name)
            plt.close()
        """
            
        # show tildeA, every bootstrap
        """
        for l in range(2):
            for bootstrap_id in range(B):
                plt.figure(figsize = (13,3))
                for i in range(n_pairs):
                    _= plt.subplot(m1,m2,i+1)
                    _= plt.imshow(tilda_A_entry[l,bootstrap_id+1,i], interpolation = "none", aspect = "auto", 
                               extent = [times_in_ms[0], times_in_ms[-1], 
                            times_in_ms[0], times_in_ms[-1]],origin= "lower", vmin = None, vmax = None)
                    _=plt.ylabel(label_names[pairs[i][0]] + " time (ms)"); 
                    plt.xlabel(label_names[pairs[i][1]] + "time (ms)")
                    _=plt.colorbar()
                    #_= plt.xlabel('ms')
                    #_= plt.ylabel('ms')
                plt.tight_layout()
                fig_name = fig_outdir + "%s_%s_%s_bt%d.pdf" %(subj, method_string[l], 'tildeA', bootstrap_id+1)
                plt.savefig(fig_name)
                plt.close()
        """
        
        
        
        #======== 20160720 added,  stability of A
        """
        # stability of A
        # get the modulus of the eigenvalues for A_cell[:,0]
        max_abs_eig = np.zeros([2,T])
        for l0 in range(2):
            for t in range(T):
                max_abs_eig[l0,t] = np.abs(np.linalg.eig(A_est[l0][t])[0]).max() 
        plt.plot(times_in_ms[1::], max_abs_eig.T)
        # variance explained, see another file
        """
    
    
    if False:
        # tildaA_all
        #tilda_value = tildaA_Z_all
        tilda_value = tildaA_all
        #tilda_value = np.abs(tildaA_Z_all)
        tildaA_T_across_subj = tilda_value.mean(axis = 1)/np.std(tilda_value.std(axis = 1))*np.sqrt(n_subj) 
        vmin, vmax = 0, None   
        for l in range(2):
            plt.figure(figsize = figsize)
            _= plt.imshow(np.abs(tildaA_T_across_subj[l]), interpolation = "none", aspect = "auto", 
                       extent = [times_in_ms[0], times_in_ms[-1], 
                    times_in_ms[0], times_in_ms[-1]],origin= "lower", vmin = vmin, vmax = vmax)
            _=plt.ylabel(label_names[pairs[pair_id][0]]+ " time (ms)"); 
            _= plt.xlabel(label_names[pairs[pair_id][1]]+" time (ms)")
            _=plt.colorbar()
            plt.tight_layout()
            
        # plot 
        #A_value = A_est_all
        A_value = A_est_all/A_se_all
        
        A_value_mean = A_value.mean(axis = 1)
        A_value_se = A_value.std(axis = 1)/np.sqrt(n_subj)
        
        ymin = None
        ymax = None
        alpha = scipy.stats.norm.ppf(1- 0.05/2)
        for l in range(2):
            plt.figure()
            count = 0
            for l1 in range(p):
                for l2 in range(p):
                    _= plt.subplot(p,p, count+1);
                    # first dim is method
                    _= plt.errorbar(times_in_ms[1::], A_value_mean[l,:,l1,l2], alpha*A_value_se[l,:,l1,l2])
                    _= plt.xlabel('time (ms)')
                    _= plt.title('A[:,%d,%d]'% (l1,l2))
                    _= plt.plot(times_in_ms[1::], np.zeros(T), 'k')
                    _= plt.ylim(ymin, ymax)
                    count += 1
            #fig_name = fig_outdir + "%s_%s_A.pdf" %(subj, method_string[l])
            plt.tight_layout()
            #plt.savefig(fig_name)
            #plt.close('all')
            


#======================= visualization ===========================================
import scipy.io
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
n_btstrp = 10

path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
#from Stat_Utility import bootstrap_mean_array_across_subjects


outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
fig_outdir ="/home/ying/Dropbox/Thesis/Dissertation/Draft/Figures/Result_figures/Scene_MEEG/source_conn/"
fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
outname_suffix = "6ROI_ico40_%s" % (fname_suffix)
p = 6

subj_list = range(1,10) + range(11,13) + range(14, 19)
n_subj = len(subj_list)
n_times = 95

n_method = 2
ROI_names = ['EVC','PPA','TOS','RSC','LOC','mOFC'] 

methods = ['dSPM','ss']

A_hat = np.zeros([n_subj, n_method, n_btstrp, n_times, p,p])
for i in range(n_subj):
    subj = "Subj%d" %subj_list[i]
    for l in range(n_btstrp):
        outname_ks = outdir + "%s_%s_MEG_ks_sol_bootstrp%d.mat" %(subj, outname_suffix,l)
        outname_dSPM = outdir + "%s_%s_MEG_dSPM_sol_bootstrp%d.mat" %(subj, outname_suffix,l)
        out_names = [outname_dSPM, outname_ks]
        for k in range(n_method):
            mat_dict = scipy.io.loadmat(out_names[k])
            A_hat[i,k,l] = mat_dict['A_hat']



A_hat0 = A_hat[:,:,0]
A_hat_se = np.std(A_hat[:,:,1::], axis = 2)

Z_stat = A_hat0/A_hat_se
logp = -np.log10(2*(1.0-scipy.stats.norm.cdf(np.abs(Z_stat))))
max_logp = 4
logp[ logp> max_logp]= max_logp
logp_mean = np.mean(logp, axis = 0)

Fisher_logp = np.zeros([n_method, n_times, p,p])
for k in range(n_method):
    for t in range(n_times):
        for j1 in range(p):
            for j2 in range(p):
                Fisher_logp[k,t,j1,j2] = -np.log10(Fisher_method(10.0**(-logp[:,k,t,j1,j2])))

max_logp2 = 15.0
Fisher_logp[ Fisher_logp > max_logp2]= max_logp2

times = (np.arange(-0.05, 0.9, 0.01)-0.04)*1000.0
import matplotlib.pyplot as plt

"""
col = ['b','g']
i0 = 15
plt.figure( figsize = (20,20) )
for i in range(6):
    for j in range(6):
        _ = plt.subplot(6,6,i*6+j+1)
        for k in range(n_method):
            _ = plt.errorbar(times, A_hat0[i0,k,:,i,j], yerr = A_hat_se[i0,k,:,i,j])
            #_ = plt.plot(times, A_hat[i0,k,:,:,i,j].T, col[k])
            #_ = plt.plot(times, logp[i0,k,:,i,j].T, col[k])
            
        _ = plt.plot(times, np.zeros(n_times), 'k')
        _ = plt.title("%s-> %s" %(ROI_names[j],ROI_names[i]))
        #_ = plt.ylim([0,10])
        #_ = plt.ylim([-0.5,0.5])
    
_= plt.tight_layout()
"""

#========= Fisher's method was very unstable ==================

#========= binomial tests, count the number of subjects above a threshold
p_binomial = np.zeros([n_method, n_times, p,p])
p0 = 0.05
thresh = -np.log10(0.01)
for k in range(n_method):
   for i in range(6):
       for j in range(6):
           tmp_logp = logp[:,k,:,i,j]
           tmp = (tmp_logp>thresh).sum(axis = 0)
           p_binomial[k,:,i,j] = -np.log10(1-scipy.stats.binom.cdf(tmp, n_subj, p0))

p_binomial[p_binomial> max_logp2] = max_logp2
# Bonferoni
bonferoni_thresh = -np.log10(0.05/n_times/p/p)



figsize = (20,15)
import matplotlib.pyplot as plt

data_name = ["binomtest","meanlogp"]
for mode in [0,1]:
    plt.figure( figsize = figsize )
    for i in range(6):
        for j in range(6):
            _ = plt.subplot(6,6,i*6+j+1)
            for k in range(n_method):
                #_ = plt.plot(times, np.abs(Z_stat[0,k,:,i,j]))
                #_ = plt.plot(times, Fisher_logp[k,:,i,j])
                if mode == 0:
                    _ = plt.plot(times, p_binomial[k,:,i,j])
                    _ = plt.plot(times, np.ones(n_times)*bonferoni_thresh, 'k')
                    _ = plt.ylim(0,max_logp2*1.05)
                elif mode == 1:
                    _ = plt.plot(times, logp_mean[k,:,i,j])
                    _ = plt.ylim(0,max_logp*1.2)
                    
            _ = plt.xlabel("time(ms)")   
            _ = plt.title("%s-> %s" %(ROI_names[j],ROI_names[i])) 
    
    _= plt.legend(methods)    
    _= plt.tight_layout()
    _= plt.savefig(fig_outdir + "6ROI_MEG_%s.pdf" %data_name[mode])
    _= plt.savefig(fig_outdir + "6ROI_MEG_%s.eps" %data_name[mode])


# ==== imshow logp====================
for k in range(n_method):
    fig = plt.figure( figsize = figsize )
    for i in range(6):
        for j in range(6):
            _ = plt.subplot(6,6,i*6+j+1)
            _ = plt.subplots_adjust(hspace =0.8 ,wspace = 0.3)
                #_ = plt.plot(times, np.abs(Z_stat[0,k,:,i,j]))
                #_ = plt.plot(times, Fisher_logp[k,:,i,j])
            im= plt.imshow(logp[:,k,:,i,j], vmin = 0, vmax = max_logp,
                           extent = [times[0], times[-1], 0, n_subj], origin = "lower",
                            interpolation = "none", aspect = "auto")
            _ = plt.title("%s-> %s" %(ROI_names[j],ROI_names[i]))
            #_ = plt.colorbar()
            _ = plt.xlabel("time (ms)")
            _ = plt.ylabel("participant id")
     
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.15])
    fig.colorbar(im, cax=cbar_ax)   
    #_= plt.tight_layout()
    _= plt.savefig(fig_outdir + "6ROI_MEG_log10p_%s.pdf" %methods[k])


"""    
# ====== log10p baseline, does not work well
btstrp_alpha = 0.05/n_times/6/6
threshold = 1.0
cluster_p_thresh = 0.05
for k in range(n_method):
    plt.figure( figsize = figsize)
    for i in range(6):
        for j in range(6):
            if i != j:
                tmp_logp = logp[:,k,:,i,j]
                #tmp_data = (tmp_logp.T - np.mean(tmp_logp[:,times<= -50.0], axis = 1)).T
                tmp_data = tmp_logp -1.0
                ax = plt.subplot(6,6,i*6+j+1) 
                tmp = bootstrap_mean_array_across_subjects(tmp_logp, alpha = btstrp_alpha)
                tmp_mean = tmp['mean']
                tmp_se = tmp['se']
                ub = tmp['ub']
                lb = tmp['lb'] 
                _ = ax.plot(times, tmp_mean)
                _ = ax.fill_between(times, ub, lb, alpha=0.4)
                _ = plt.ylim(0,4)
                
                #_ = plt.tight_layout(0.001)    
                Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(tmp_data, threshold,tail = 0)
                print clusters, p_val_clusters
                tmp_window = list()
                count0 = 4
                count = 0
                for i_c, c in enumerate(clusters):
                    c = c[0]
                    text_y = np.array([0.3,-0.3,0.4, -0.4])
                    if p_val_clusters[i_c] <= cluster_p_thresh:
                        print count
                        count = count+1
                        _ = ax.axvspan(times[c.start], times[c.stop - 1],
                                            color='k', alpha=0.1)
                        print count, l, text_y[np.mod(count,3)]     
                        _ = plt.text(times[c.start],text_y[np.mod(count,3)],('p = %1.3f' %(p_val_clusters[i_c])))
                        tmp_window.append(dict(start = c.start, stop = c.stop, p = p_val_clusters[i_c]))
                
        
            _ = plt.title("%s-> %s" %(ROI_names[j],ROI_names[i]))
            _ = plt.xlabel("time (ms)")
        
    _= plt.tight_layout()
    _= plt.savefig(fig_outdir + "6ROI_MEG_meanlog10p_%s.pdf" %methods[k])
"""

                