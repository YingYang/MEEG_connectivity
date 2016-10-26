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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#matplotlib.use('Agg')                    
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
#from get_simu_data_ks import get_simu_data_ks
from ROI_Kalman_smoothing import get_cov_u

subj_list = [1,2,3,4,5,6,7,8,9,10,12,13]
n_subj = len(subj_list)
outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
pairs = [[0,0],[1,1],[0,1]]
#pairs = [[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]]
m1,m2 = 1,3
n_pairs = len(pairs)
    
B = 10
mne_method = 'MNE'
ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g'] #, 'LO_c_g']
#for i0 in range(n_subj):
print "B = %d" %B


p = len(ROI_bihemi_names)

method_string = ['ks','dspm']
flag_cov_from_u = False
fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/roi_ks/"  

label_names = ['EVC','PPA']
p = len(label_names)
if True:
    i0 = 7
    subj = "Subj%d" %subj_list[i0]
    print subj
    
    #for i0 in range(n_subj):
    #subj = "Subj%d" %subj_list[i0]  
    
    # load the time indices
    ave_mat_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"\
                     + "%s/%s_%s" %(subj,  subj, "1_110Hz_notch_ica_ave_alpha15.0.mat")
    mat_dict = scipy.io.loadmat(ave_mat_path)
    times = mat_dict['times'][0]
    del(mat_dict)
    offset = 0.04
    time_ind = np.all(np.vstack([times >= -0.06, times <= 0.74 ]), axis = 0)
    times_in_ms = (times[time_ind]-offset)*1000.0
    print len(times_in_ms)    
    T = len(times_in_ms)-1
    
 
    A_cell = np.zeros([2,B+1,T,p,p])
    lagged_corr = np.zeros([2,B+1,n_pairs,T+1,T+1])
    lagged_inv = np.zeros([2, B+1, n_pairs,T+1,T+1])
    std_u = np.zeros([2, B+1,T+1,p])
    tilda_A_entry = np.zeros([2,B+1,n_pairs, T,T])
    for l in range(2):
        for bootstrap_id in range(B+1):
            out_name = outdir + "%s_%s_sol_bootstrp%d.mat" % (subj, method_string[l],bootstrap_id)
            result = scipy.io.loadmat(out_name)
            A_cell[l,bootstrap_id, :,:,:] = result['A_hat']
            u_array_hat = result['u_array_hat']
            std_u[l,bootstrap_id] = np.std(u_array_hat, axis = 0)
            
            if flag_cov_from_u:       
                cov_u = np.cov(u_array_hat.reshape([u_array_hat.shape[0], -1]).T)    
            else:
                cov_u = get_cov_u(result['Q0_hat'], result['A_hat'], result['Q_hat'], T, 
                                  flag_A_time_vary = True) # pT x pT, p first 
            paired_lags = np.zeros([n_pairs, (T+1), (T+1)])
            for i in range(n_pairs):
                for t1 in range(T+1):
                    for t2 in range(T+1):
                        paired_lags[i,t1,t2] = cov_u[t1*p+pairs[i][0], t2*p+pairs[i][1]]\
                        /np.sqrt(cov_u[t1*p+pairs[i][0], t1*p+pairs[i][0]])/ np.sqrt(cov_u[t2*p+pairs[i][1], t2*p+pairs[i][1]])
                lagged_corr[l, bootstrap_id] = paired_lags            
            inv_cov_u = np.linalg.inv(cov_u)
            #plt.figure(); plt.imshow(inv_cov_u, interpolation = "none"); plt.colorbar();
            paired_lags_inv = np.zeros([n_pairs, (T+1), (T+1)])
            for i in range(n_pairs):
                for t1 in range(T+1):
                    for t2 in range(T+1):
                        paired_lags_inv[i,t1,t2] = inv_cov_u[t1*p+pairs[i][0], t2*p+pairs[i][1]]
                lagged_inv[l, bootstrap_id] = paired_lags_inv
            
            
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
            for i in range(n_pairs):
                for t1 in range(T):
                    for t2 in range(T):
                        if t1<= t2:
                            #print t1, t2,i,n_pairs
                            #print tilde_A[t1,t2]
                            #print pairs[i][1], pairs[i][0]
                            # region 1-> region 2  upper diagonal leadding
                            tilda_A_entry[l,bootstrap_id, i,t1,t2] = tilde_A[t1,t2][pairs[i][1], pairs[i][0]]
                        else:
                            # region 2 -> region 1 lower diangonal feedback
                            tilda_A_entry[l,bootstrap_id, i,t1,t2] = tilde_A[t2,t1][pairs[i][0], pairs[i][1]]
            del(result)
            
    
    # plot the all parts in A, with bootstrap CI
    print "plotting A"
    A_est = A_cell[:,0]
    A_se = np.std(A_cell[:,1::], axis = 1)
    
    var_u_est = std_u[:,0]**2
    var_u_se = np.std(std_u[:,1::]**2, axis = 1)
    
    print "std_u.shape"
    print std_u.shape
    print var_u_est.shape
    print var_u_se.shape
    
    alpha0 = scipy.stats.norm.ppf(1- 0.05/2)
    for l in range(2):
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
        plt.legend(label_names)
        fig_name = fig_outdir + "%s_%s_var_u.pdf" %(subj, method_string[l])
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close('all')
        
    
    print A_est.shape
    print A_se.shape
    print A_est[0,2::2,0,0]
    print A_se[0,2::2,0,0]

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
                _= plt.errorbar(times_in_ms[1::], A_est[l,:,l1,l2], alpha*A_se[l,:,l1,l2])
                _= plt.xlabel('time (ms)')
                _= plt.title('A[:,%d,%d]'% (l1,l2))
                _= plt.plot(times_in_ms[1::], np.zeros(T), 'k')
                _= plt.ylim(ymin, ymax)
                count += 1
        fig_name = fig_outdir + "%s_%s_A.pdf" %(subj, method_string[l])
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close('all')
        
    # just plot A_est alone
    for l in range(2):
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
                fig_name = fig_outdir + "%s_%s_%s_thresh%d.pdf" %(subj, method_string[l], lag_names[j], flag_thresh)
                plt.savefig(fig_name)
                plt.close()
    
    #T_thresh = np.abs(scipy.stats.norm.ppf(0.1/2.0/np.float(T+1)**2)) 
    # only do the relevant pairs
    alpha = 0.05
    pair_id = 2    
    vmin, vmax = 0, 10
    figsize = (4.6,4)
    fdr_flag = False
    for flag_thresh in [True, False]:
        for l in range(2):
            print "saving tilde A"          
            tmp = tilda_A_entry
            est = tmp[l,0,pair_id]
            se = np.std(tmp[l,1::, pair_id,:,:], axis = 0)
            tmp_T = est/se
            tmp_T[np.isnan(tmp_T)] = 0
            tmp_T[np.isinf(tmp_T)] = 0
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
                    T_thresh = np.abs(scipy.stats.norm.ppf(alpha/2.0/np.float(T+1)**2))
                    tmp_T1 = np.abs(tmp_T)
                    tmp_T1[tmp_T1<=T_thresh] = 0
            else:
                tmp_T1 = np.abs(tmp_T)
                
            plt.figure(figsize = figsize)
            _= plt.imshow(tmp_T1, interpolation = "none", aspect = "auto", 
                       extent = [times_in_ms[0], times_in_ms[-1], 
                    times_in_ms[0], times_in_ms[-1]],origin= "lower", vmin = vmin, vmax = vmax)
            _=plt.ylabel(label_names[pairs[i][0]]+ " time (ms)"); 
            _= plt.xlabel(label_names[pairs[i][1]]+" time (ms)")
            _=plt.colorbar()
            plt.tight_layout()
            fig_name = fig_outdir + "%s_%s_%s_thresh%d_fdr%d.pdf" %(subj, method_string[l], 'tildeA', flag_thresh, fdr_flag)
            plt.savefig(fig_name)
            plt.close()
    
    # just show the estimated tildeA alone   
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
        
    # show tildeA, every bootstrap
    # just show the estimated tildeA alone
    if False:
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
    


            
           
   