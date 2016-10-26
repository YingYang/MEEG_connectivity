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
plt.rcParams['image.cmap'] = 'plasma'
#from get_simu_data_ks import get_simu_data_ks
from ROI_Kalman_smoothing import get_cov_u



import numpy as np
import sys
import scipy.io

import mne
from mne.forward import  is_fixed_orient, _to_fixed_ori
from mne.inverse_sparse.mxne_inverse import _prepare_gain
import copy


def save_visualized_jt(M, noise_cov_path,  evoked_path, 
                       Sigma_J_list, ut,
                       ROI_list, n_ROI_valid,
                       subjects_dir,
                       subj,                       
                       fwd_path, 
                       out_stc_name, out_fig_name,
                       whiten_flag, depth = None, force_fixed=True, tmin= 0, tstep=0.01):
    """
    # ut and yt can be for a single time point
    """
    
    if depth == None:
        depth = 0.0
    
    q,n,T = M.shape
    # this function returns a list, take the first element
    evoked = mne.read_evokeds(evoked_path)[0]
    # depth weighting, TO BE MODIFIED
    print force_fixed
    fwd0 = mne.read_forward_solution(fwd_path, force_fixed= force_fixed, surf_ori = True)
    fwd= copy.deepcopy(fwd0)
    noise_cov = mne.read_cov(noise_cov_path)
    Sigma_E = noise_cov.data
    
    # orientation of dipoles
    ind0 = fwd['src'][0]['inuse']
    ind1 = fwd['src'][1]['inuse']
    # positions of dipoles
    rr = np.vstack([fwd['src'][0]['rr'][ind0==1,:], 
                             fwd['src'][1]['rr'][ind1==1,:]])
    rr = rr/np.max(np.sum(rr**2, axis = 1))                       
    nn = np.vstack([fwd['src'][0]['nn'][ind0 == 1,:],
                    fwd['src'][1]['nn'][ind1 == 1,:]])
    # number of dipoles                
    m = rr.shape[0]
    all_ch_names = evoked.ch_names
    sel = [l for l in range(len(all_ch_names)) if all_ch_names[l] not in evoked.info['bads']]
    
    if force_fixed:
        print "fixed orient"
        print is_fixed_orient(fwd)
        if not is_fixed_orient(fwd):
            _to_fixed_ori(fwd)
        print "difference of G"
        print np.max(np.abs(fwd['sol']['data']))/np.min(np.abs(fwd['sol']['data']))
    
    if whiten_flag:
        pca = True
        G, G_info, whitener, source_weighting, mask = _prepare_gain(fwd, evoked.info,
                                                                    noise_cov, pca =pca,
                                                                    depth = depth, loose = None,
                                                                    weights = None, weights_min = None)
        #Sigma_E_chol = np.linalg.cholesky(Sigma_E)
        #Sigma_E_chol_inv = np.linalg.inv(Sigma_E_chol)
        #G = np.dot(Sigma_E_chol_inv, G)
        # after whitening, the noise cov is assumed to identity
        #Sigma_E = (np.dot(Sigma_E_chol_inv, Sigma_E)).dot(Sigma_E_chol_inv.T)
        Sigma_E = np.eye(G.shape[0])
        M = (np.dot(whitener, M)).transpose([1,0,2])
    else:
        G = fwd['sol']['data'][sel,:]
        G_column_weighting = (np.sum(G**2, axis = 0))**(depth/2)
        G = G/G_column_weighting
        
   
    QJ = np.zeros(m)
    for l in range(len(ROI_list)):
        QJ[ROI_list[l]] = Sigma_J_list[l]
        
    L = np.zeros([m,n_ROI_valid])
    for l in range(n_ROI_valid):
        L[ROI_list[l],l] = 1.0
        
    # compute the inverse
    inv_Sigma_E = np.linalg.inv(Sigma_E)
    GQE = G.T.dot(inv_Sigma_E)
    GQEG = GQE.dot(G)
    QJ_inv = 1.0/QJ
    GQEG += np.diag(QJ_inv)
    inv_op = np.linalg.inv(GQEG)
    
    QJL = (L.T/QJ).T
    
    
    J = np.zeros([q, m, T])
    for r in range(q):
        J[r] = inv_op.dot(np.dot(GQE, M[r]) + np.dot(QJL, ut[r]))
        
    
    
     # mne results
    evoked = mne.read_evokeds(evoked_path)[0]
    # depth weighting, TO BE MODIFIED
    noise_cov = mne.read_cov(noise_cov_path)
                    
    ch_names = evoked.info['ch_names']
    # create the epochs first?
    M_all = np.zeros([q, len(ch_names), T])
    valid_channel_ind = [i for i in range(len(ch_names)) if ch_names[i] not in evoked.info['bads'] ]
    M_all[:,valid_channel_ind, :] = M.copy()    
    events = np.ones([M.shape[0],3], dtype = np.int)                
    epochs = mne.EpochsArray(data = M_all, info = evoked.info, events = events,
                             tmin =  evoked.times[0], event_id = None, reject = None)
    method = "MNE"
    lambda2 = 1.0
    depth0 = None if depth == 0 else depth
    inv_op = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov,
               loose = 0.0, depth = depth0,fixed = True)
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2 = lambda2,
                 method = method)
        
    J_mne = np.zeros([q, m, T])
    for r in range(q):
        J_mne[r] = stcs[r].data

    
    
    
     # 
    # compute the std of J
    J_std = np.std(J, axis = 0)
    u_std = np.std(ut, axis = 0)
    J_mne_std = np.std(J, axis = 0)
    
    trial_ind = 0
    tmp_J_list = [J[trial_ind], J_mne_std[trial_ind], J_std]
    tmp_u_list = [ut[trial_ind], ut[trial_ind], u_std]
    suffix_list = ['trial%d' % trial_ind, "mne", "std"]
    # some other visulaizaation
    times_in_ms = (np.arange(tmin, tmin+T*tstep, tstep))*1000.0
    
    for ll in range(2):
        tmp_J, tmp_u, suffix = tmp_J_list[ll], tmp_u_list[ll], suffix_list[ll]
        plt.figure()
        for l in range(n_ROI):
            ROI_id = 0
            _= plt.subplot(n_ROI, 1, l+1)
            _= plt.plot(times_in_ms, tmp_J[ROI_list[l],:].T, 'b', alpha = 0.1)
            if l < n_ROI_valid:
                _= plt.plot(times_in_ms, tmp_u[l, :], 'k', lw = 2, alpha = 1)
                ROI_id = l+1
                
            _= plt.xlabel('time ms')
            _= plt.title("ROI %d" %ROI_id)
        _= plt.tight_layout()
        _ = plt.savefig(out_fig_name + "%s.pdf" %suffix)
              
    


   

    
    
   
    # save as an STC
    vertices_to = [fwd['src'][0]['vertno'], 
                   fwd['src'][1]['vertno']]
    stc = mne.SourceEstimate(data = J_std,vertices = vertices_to, 
                                    tmin = tmin, tstep = tstep )
    stc.save(out_stc_name) 
    
    # render the images

    clim = dict(kind='value', lims=np.array([0.1, 2, 10])*1E-10)
    time_seq = np.arange(0, T, 10)
    surface = "inflated"
   
    brain = stc.plot(surface= surface, hemi='both', subjects_dir=subjects_dir,
                    subject = subj,  clim=clim)
    for k in time_seq:
        brain.set_data_time_index(k)
        for view in ['ventral']:
            brain.show_view(view)
            im_name = out_fig_name + "%03dms_%s.pdf" \
               %(np.int(np.round(stc.times[k]*1000)), view)
            brain.save_image(im_name) 
            print k
    brain.close()
    
    for hemi in ['lh','rh']:
        brain = stc.plot(surface=surface, hemi= hemi, subjects_dir=subjects_dir,
                subject = subj,  clim=clim)
        for k in time_seq:
            brain.set_data_time_index(k)
            for view in ['medial','lateral']:
                brain.show_view(view)
                im_name = out_fig_name + "%03dms_%s_%s.pdf" \
               %(np.int(np.round(stc.times[k]*1000)), view, hemi)
                brain.save_image(im_name)          
        brain.close()
    
    return 0
    
  
if __name__ == "__main__":
    
    #flag_real_data = False
    flag_real_data = True
    
    if flag_real_data:
        #================= setting paths
        MEGorEEG = ['EEG','MEG']
        # test Subject 8
        outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/nips2016_submission/"
        labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/" 
        MEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
        EEG_data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
        
        # to be changed to the correct one
        MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
        EEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
    
        isMEG = True
        fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
        flag_ico_4 = True
        outname_suffix = "EVC_PPA_ico4%d_%s" % (flag_ico_4, fname_suffix)
        
        subj = "Subj8"
        subjects_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
    
        if isMEG:
            if not flag_ico_4:
                fwd_path = MEG_data_dir + "fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            else:
                fwd_path = MEG_data_dir + "fwd/%s/%s_ico-4_run1-fwd.fif" %(subj, subj)
                
            epochs_path = MEG_data_dir + "epoch_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica-epo.fif.gz" %(subj,subj)
            ave_mat_path =  MEG_data_dir + "epoch_raw_data/%s/%s_%s.mat" %(subj,  subj, fname_suffix)
        else:
            if not flag_ico_4:
                fwd_path = EEG_data_dir + "%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
            else:
                fwd_path = EEG_data_dir + "%s_EEG/%s_EEG_ico-4-fwd.fif" %(subj, subj)
                  
            epochs_path = EEG_data_dir + "epoch_raw_data/%s_EEG/%s_EEG_filter_1_110Hz_notch_ica_reref-epo.fif.gz" %(subj,subj)
            ave_mat_path =  EEG_data_dir + "epoch_raw_data/%s_EEG/%s_EEG_%s.mat" %(subj,  subj, fname_suffix)          
                                           
        #=============================
        # temporary files for quicker computation
        # evoked template and noise covariance
        #evoked_path = outdir+ "%s_%s_tmp_evoked-ave.fif.gz" %(subj, MEGorEEG[isMEG])  
        #noise_cov_path =  outdir+"%s_%s_noise_cov-cov.fif" %(subj, MEGorEEG[isMEG])
        method = "dspm"
        noise_cov_path = outdir + "%s_STFT-R_MEGnoise_cov-cov.fif" %subj
        evoked_path = outdir + "%s_tmp_evoked-ave.fif.gz" %subj
        sol_path = outdir + "%s_%s_sol_bootstrp0.mat" %(subj, method)
        out_stc_name = outdir + "%s_%s_sol_jt_std" %(subj, method)
        out_fig_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/roi_ks/source_visualization/%s_%s_nips" %(subj, method)
        whiten_flag = True if method == "ks" else False
        tmin, tmax = -0.06, 0.74
        tstep = 0.01
        force_fixed = True if method == "ks" else False
        #================== define the ROIs=============================                   
        # read ROI labels
        ROI_bihemi_names = [ 'pericalcarine', 'PPA_c_g']
        labeldir1 = labeldir + "%s/labels/" % subj
        # load and merge the labels
        labels_bihemi = list()
        lr_label_list = list()
        lr_label_names = list()
        for j in range(len(ROI_bihemi_names)):
            for hemi in ['lh','rh']:
                print subj, j, hemi
                tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj, ROI_bihemi_names[j],hemi)
                tmp_label = mne.read_label(tmp_label_path)
                lr_label_list.append(tmp_label)
                lr_label_names.append(ROI_bihemi_names[j]+"_" + hemi)
        for j in range(len(ROI_bihemi_names)):
            labels_bihemi.append(lr_label_list[2*j]+lr_label_list[2*j+1]) 
        
        flag_merge_bilateral = True
        labels = labels_bihemi if flag_merge_bilateral else lr_label_list
        label_names = ROI_bihemi_names if flag_merge_bilateral else lr_label_names
        # keep fixed orientation                 
        fwd = mne.read_forward_solution(fwd_path, force_fixed=True, surf_ori=True)
        m = fwd['sol']['ncol']
        
        src = fwd['src']
        ROI_list0 = list() 
        for i in range(len(labels)):
            _, sel = mne.source_space.label_src_vertno_sel(labels[i],src)
            ROI_list0.append(sel)
        
        ROI_list = copy.deepcopy(ROI_list0)
        # check if the ROIs has overlap, if so, move the overlap to non-ROI 
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                tmp_intersect = np.intersect1d(ROI_list0[i], ROI_list0[j])
                if len(tmp_intersect) > 0:
                    print "intersection between %d and %d found, removing it" %(i,j)
                    print tmp_intersect
                    ROI_list[i] = np.setdiff1d(ROI_list0[i], tmp_intersect)
                    ROI_list[j] = np.setdiff1d(ROI_list0[j], tmp_intersect)
            
        ROI0_ind = np.arange(0, m, dtype = np.int)    
        for i in range(len(labels)):
            ROI0_ind = np.setdiff1d(ROI0_ind, sel)
        ROI_list.append(ROI0_ind)
        n_ROI = len(ROI_list)
        n_ROI_valid = n_ROI-1
    
        #  ============prep operations to for the data=============================
        n_im = 362           
        mat_dict = scipy.io.loadmat(ave_mat_path)
        data = mat_dict['ave_data']
        picks_all = mat_dict['picks_all'][0]
        times = mat_dict['times'][0]
        data -= np.mean(data, axis = 0)
        del(mat_dict)
    
        #offset = 0.04 if isMEG else 0.00
        # nips version
        time_ind = np.all(np.vstack([times >= tmin, times <= tmax ]), axis = 0)
        M = data[:,:, time_ind]
       
        result = scipy.io.loadmat(sol_path)
        ut = result['u_array_hat'].transpose([0,2,1])
        Sigma_J_list = result['Sigma_J_list_hat'][0]
        
        
        save_visualized_jt(M, noise_cov_path,  evoked_path, 
                           Sigma_J_list, ut,
                           ROI_list, n_ROI_valid,
                           subjects_dir,
                           subj,                       
                           fwd_path, 
                           out_stc_name, out_fig_name,
                           whiten_flag, depth = None, force_fixed=True, tmin= tmin, tstep=tstep)
    
    
    else:
        #================= simulations ============================================
        anat_ROI_names= ['pericalcarine-lh', 'pericalcarine-rh',
                     #'lateraloccipital-lh', 'lateraloccipital-rh',]
                    'parahippocampal-lh', 'parahippocampal-rh',]

        alpha = 2
        flag_random_A = False
        flag_time_smooth_source_noise = True
        flag_space_smooth_source_noise = False
        flag_nn_dot = True
        whiten_flag = True
        
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
        
    

        
        simu_path = "/home/ying/dropbox_unsync/MEEG_source_roi_cov_simu_and_data/"
        outpath = simu_path + \
                   "/%s_ROI_alpha%1.1f_simu%d_randA%d_t%d_s%d_nn%d" \
                   %(p,alpha, 0, flag_random_A, 
                      flag_time_smooth_source_noise, 
                      flag_space_smooth_source_noise,
                      flag_nn_dot)
        
        mat_dict = scipy.io.loadmat(outpath + ".mat")
        
        fwd_path = mat_dict['fwd_path'][0]
        noise_cov_path = mat_dict['noise_cov_path'][0]
        
        ROI_list = list()
        n_ROI = len(mat_dict['ROI_list'][0])
        n_ROI_valid = n_ROI-1
        for i in range(n_ROI):
            ROI_list.append(mat_dict['ROI_list'][0,i][0])
        M = mat_dict['M']
        
        
        result = scipy.io.loadmat(outpath+"_ks_sol.mat")
        ut = result['u_array_hat'].transpose([0,2,1])
        Sigma_J_list = result['Sigma_J_list_hat'][0]
        
        evoked_path = outpath+"-ave.fif.gz"
    
    

