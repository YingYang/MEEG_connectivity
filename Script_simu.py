# -*- coding: utf-8 -*-
import numpy as np
import mne
import sys
import scipy.io
import copy



# this part to be optimized? chang it to a package?
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)
from ROI_cov_Kronecker import sample_kron_cov                        
import matplotlib.pyplot as plt
from get_simu_data import get_simu_data
from get_estimate import get_estimate



flag_simu = False
flag_ROIcov = False
flag_ROIcovKronecker = False
flag_mne = True
flag_mneKronecker = True

anat_ROI_list_fname = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/aparc_ROI_names.npy"
#========
#data_path = mne.datasets.sample.data_path()
#subjects_dir = data_path + '/subjects'
#labels = mne.read_labels_from_annot('sample', parc='aparc',
#                                    subjects_dir=subjects_dir)
#ROI_names = list()
#for label in labels:
#    ROI_names.append(label.name)
#np.save(ana_ROI_list_fname, ROI_names)
#=================================
#anat_ROI_names = np.load(anat_ROI_list_fname)
#p = 10
#anat_ROI_names = anat_ROI_names[np.random.choice(range(len(anat_ROI_names)), p, replace = False)]
p = 6
anat_ROI_names= ['lateralorbitofrontal-lh','lateralorbitofrontal-rh',
                 'parahippocampal-lh', 'parahippocampal-rh',
                 'lateraloccipital-lh', 'lateraloccipital-rh']

simu_id = int(sys.argv[1])
L_list_option = int(sys.argv[2])
# 0 is all one, 1 is sign flip, 2 is svd, 3 is spatial smooth
print simu_id
print type(simu_id)

outdir = "/home/ying/dropbox_unsync/MEEG_source_roi_cov_simu_and_data/roi_cov_kronecker/"
outpath = outdir + "%d_ROIs_L%d_simu%d.mat" %(p, L_list_option, simu_id)
simufilepath = outpath

if flag_simu:
    np.random.seed()
    
    #========== QUcov=======
    alpha = 5.0
    p = len(anat_ROI_names)
    tmp = np.random.randn(p,p)
    r = np.random.gamma(shape=0.5, scale=1.0, size=p)
    QUcov = np.dot(tmp*r, (tmp*r).T)
    QUcov += np.eye(p)
    diag = np.sqrt(np.diag(QUcov))
    denom = np.outer(diag, diag)
    QUcov = QUcov/denom* alpha
    
    scale_factor = 1E-9
    QUcov = QUcov*scale_factor**2
    #=========Tcov=======
    T = 5
    a0,b0 = 1.0, 1E-1 # a exp (-b ||x-y||^2)
    # Gaussian process kernel for temporal smoothness
    Tcov = np.zeros([T,T])
    for i in range(T):
        for j in range(T):
            Tcov[i,j] = a0 * np.exp(-b0 * (i-j)**2)
    Tcov += 0.01*np.eye(T)
    print np.linalg.cond(Tcov)
    
    #========Sigma_J_list ===
    Sigma_J_list = np.random.gamma(shape=2, scale=1.0, size= p+1)
    Sigma_J_list = Sigma_J_list*scale_factor**2
    #========
    # generate the simulation data
    q = 320
    cov_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/" \
                        + "Subj1_STFT-R_all_image_Layer_1_7_CCA_ncomp6_MEGnoise_cov" 
    get_simu_data(q,T, anat_ROI_names, outpath,
                                QUcov, Tcov, Sigma_J_list, 
                                L_list_option = L_list_option,
                                L_list_param = None,
                                normalize_G_flag = False,
                                sensor_iir_flag = False,
                                cov_out_dir = cov_out_dir)
                            
#=======solution             
if flag_ROIcov:               
    #============ ROI cov ===============
    method = "ROIcov"
    print method
    tmp_outname = outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, method)
    get_estimate(simufilepath, tmp_outname, method = "ROIcov", 
                     loose = None, depth = None, whiten_flag = False,
                     verbose = True) 
    

if flag_ROIcovKronecker:  
    """
    whiten_flag must be true! 
    """
    print "ROI_covKronecker"
    tmp_result = scipy.io.loadmat(outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, "ROIcov"))
    #========== ROI cov Kronecker
    method = "ROIcovKronecker"
    print method
    # initialize the parameters
    Qu0 = tmp_result['Qu_hat']
    L_list0 = list()
    for i in range(p):
        L_list0.append(tmp_result['L_list_hat'][0,i][0])    
    Sigma_J_list0 = tmp_result['Sigma_J_list_hat'][0]
    
    tmp_outname = outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, method)
    get_estimate(simufilepath, tmp_outname, method = method, 
                     loose = None, depth = None, whiten_flag = True,
                     verbose = True, Qu0 = Qu0,
                 L_flag = True, L_list0 = L_list0, Sigma_J_list0 = Sigma_J_list0  )   
                 
    """
    # debug
    result_mat_dict = scipy.io.loadmat(tmp_outname)
    true_mat_dict =scipy.io.loadmat(simufilepath+".mat")
    
    plt.figure()
    plt.subplot(1,2,1); plt.imshow(true_mat_dict['Tcov'], interpolation = "none"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(result_mat_dict['Tcov_hat'], interpolation = "none"); plt.colorbar()
    
    plt.figure()
    plt.subplot(1,2,1); plt.imshow(true_mat_dict['QUcov'], interpolation = "none"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(result_mat_dict['Qu_hat'], interpolation = "none"); plt.colorbar()
    
    Qu_all = [true_mat_dict['QUcov'], result_mat_dict['Qu_hat']]
    corr_all = np.zeros([len(Qu_all), p, p])
    for l1 in range(len(Qu_all)):
        Qu_hat = Qu_all[l1]
        diag0 = np.sqrt(np.diag(Qu_hat))
        denom = np.outer(diag0, diag0)
        corr_hat = np.abs(Qu_hat/denom)
        corr_all[l1] = corr_hat
        
    plt.figure()
    for l1 in range(len(Qu_all)):
        plt.subplot(1,2,l1+1); plt.imshow(corr_all[l1], interpolation = "none"); plt.colorbar()
    """

lambda_seq = 10.0**np.arange(-3,4)                      
if flag_mne:    
    #============ MNE true L
    for method in [ "mneTrueL", "mneFlip", "mnePairwise"]:
        for l in range(len(lambda_seq)): 
            method0 = "%s_loglam%1.1f" %(method, np.log10(lambda_seq[l]))
            print method
            tmp_outname = outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, method0)
            get_estimate(simufilepath, tmp_outname, method = method, 
                         loose = None, depth = None, whiten_flag = False,
                         verbose = True, lambda2 = lambda_seq[l])

# To be verified
if flag_mneKronecker:
     for method in [ "mneTrueLKronecker", "mneFlipKronecker"]:
        for l in range(len(lambda_seq)): 
            method0 = "%s_loglam%1.1f" %(method, np.log10(lambda_seq[l]))
            print method
            tmp_outname = outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, method0)
            get_estimate(simufilepath, tmp_outname, method = method, 
                         loose = None, depth = None, whiten_flag = False,
                         verbose = True, lambda2 = lambda_seq[l])

if False:
    L_list_option = 1
    outdir = "/home/ying/dropbox_unsync/MEEG_source_roi_cov_simu_and_data/roi_cov_kronecker/"
    
    simu_id_seq = range(1,6)
    n_simu = len(simu_id_seq)
    lambda_seq = 10.0**np.arange(-3,4) 
    n_Lambda = len(lambda_seq)
    
    n_method = 7
    dist_to_true_corr_all = np.zeros([n_simu, n_method])
    corr_with_true_corr_all = np.zeros([n_simu, n_method])
    
    extra_method_seq = [ "mneTrueL", "mneFlip", "mnePairwise", "mneTrueLKronecker", "mneFlipKronecker"]
    
    for j in range(n_simu):
        simu_id = simu_id_seq[j]
        simufilepath = outdir + "%d_ROIs_L%d_simu%d.mat" %(p, L_list_option, simu_id)
        out_name = [  outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, "ROIcov")]
        out_name += [ outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, "ROIcovKronecker") ]
        for method in extra_method_seq:
            for l in range(len(lambda_seq)):
                method0 = "%s_loglam%1.1f" %(method, np.log10(lambda_seq[l]))
                print method
                tmp_outname = outdir + "%d_ROIs_L%d_simu%d_%s.mat" %(p, L_list_option, simu_id, method0)
                out_name.append(tmp_outname)
        
        # load the true Qucov
        true_mat_dict = scipy.io.loadmat(simufilepath+".mat")
        QUcov = true_mat_dict['QUcov']
        
        Qu_all = [QUcov]
        for l1 in range(len(out_name)):
            tmp_mat = scipy.io.loadmat(out_name[l1])
            if tmp_mat['Qu_hat'].shape[0] < p:
                Qu_hat = tmp_mat['corr_hat']
            else:
                Qu_hat = tmp_mat['Qu_hat']
            Qu_all.append(Qu_hat)
            
        corr_all = np.zeros([len(Qu_all), p, p])
        for l1 in range(len(Qu_all)):
            Qu_hat = Qu_all[l1]
            diag0 = np.sqrt(np.diag(Qu_hat))
            denom = np.outer(diag0, diag0)
            corr_hat = np.abs(Qu_hat/denom)
            corr_all[l1] = corr_hat
         
        #===================================
        #names = ['truth','ROIcov']
        names = ['truth','one-step t',' one-step \n Kronecker']
        for method in extra_method_seq:
            for l in range(len(lambda_seq)):
                names.append("%s \n log lambda %d" %(method, np.log10(lambda_seq[l])))
        
        """
        # check L_list
        True_L_list = true_mat_dict['L_list']
        L_list = scipy.io.loadmat(out_name[0])['L_list_hat']
        
        plt.plot(True_L_list[0][0][0], L_list[0][0][0], '.')
        
        
        
        # TrueL and Flip was the same
        fig = plt.figure(figsize = (15,3))
        to_show_ind = [0,1,2,6,20,27]
        m1,m2 = 1, len(to_show_ind)
        count = 0
        for l1 in to_show_ind:
            plt.subplot(m1,m2,count+1)
            count += 1
            im= plt.imshow(corr_all[l1], interpolation = "none", vmin = 0, vmax = 1)
            _= plt.title(names[l1])
            
        cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
        fig.colorbar(im, cax=cbar_ax)    
        plt.savefig("/home/ying/Dropbox/Thesis/Dissertation/Draft/Figures/Result_figures/roi_cov/roi_cov_L%d_abscorr.pdf" %L_list_option)
        """
        
        mask = np.ones([p,p])
        mask = np.triu(mask,1)
    
        dist_to_true_corr = np.zeros(corr_all.shape[0]-1)
        corr_with_true_corr = np.zeros(corr_all.shape[0]-1)
        for l in range(corr_all.shape[0]-1):
            dist_to_true_corr[l] = np.linalg.norm(corr_all[l+1]- corr_all[0])
            tmp_triu0 = corr_all[0][mask>0]
            for l in range(corr_all.shape[0]-1):
                tmp_triu1 = corr_all[l+1][mask>0]
                corr_with_true_corr[l] = np.corrcoef(np.vstack([tmp_triu0, tmp_triu1]))[0,1]
        print dist_to_true_corr
        print corr_with_true_corr
        
        # hard coded
        # first two methods are ROIcov, ROIcovKronecker
        n_method0 = 2 
        dist_to_true_corr_all[j,0:n_method0] = dist_to_true_corr[0:n_method0]
        corr_with_true_corr_all[j,0:n_method0] = corr_with_true_corr[0:n_method0]
        for k in range(n_method-n_method0):
            dist_to_true_corr_all[j,k+n_method0] = np.min(dist_to_true_corr[k*n_Lambda + n_method0:(k+1)*n_Lambda + n_method0])
            corr_with_true_corr_all[j,k+n_method0] = np.max(corr_with_true_corr[k*n_Lambda + n_method0:(k+1)*n_Lambda + n_method0])
    
    
    
    
    
    to_show_ind = [0,1,2,4,5]
   
    width = 0.8
    method_seq = ['onestep t', 'onestep Kronecker'] + extra_method_seq 
    xticknames_pair = []
    pairs = [[1,0],[2,0],[4,0],[5,0]]
    for j0 in range(len(pairs)):
        xticknames_pair.append("%s\n-%s" %(method_seq[pairs[j0][0]],  method_seq[pairs[j0][1]]))
        
    n_pairs = len(pairs)
    diff_dist_to_true_corr_all = np.zeros([n_simu, n_pairs])
    diff_corr_with_true_corr_all = np.zeros([n_simu, n_pairs])
    for j0 in range(len(pairs)):
        diff_dist_to_true_corr_all[:,j0] = (dist_to_true_corr_all[:,pairs[j0][0]]     - dist_to_true_corr_all[:,pairs[j0][1]])  
        diff_corr_with_true_corr_all[:,j0] = (corr_with_true_corr_all[:,pairs[j0][0]] - corr_with_true_corr_all[:,pairs[j0][1]]) 
    
    
    data_name = ['RMSE', 'corr upper tri']
    data = list()
    data = [dist_to_true_corr_all[:, to_show_ind], corr_with_true_corr_all[:, to_show_ind]]
    data_diff = [diff_dist_to_true_corr_all, diff_corr_with_true_corr_all]
    
    
    xticknames1 = [ method_seq[i] for i in to_show_ind]
    
    plt.figure(figsize = (12,8))
    
    xtick = np.arange(1, len(to_show_ind)+1)-0.3
    for j0 in range(2):
        plt.subplot(2,2,j0*2+1)
        tmp_data = data[j0]
        plt.bar(xtick,np.mean(tmp_data, axis = 0), width = width, color = 'y')
        plt.errorbar(xtick+width/2, np.mean(tmp_data, axis = 0), 
                     np.std(tmp_data,axis = 0)/np.sqrt(n_simu), fmt = None, lw = 2, ecolor = 'k')
        plt.xticks(xtick+width/2,xticknames1, rotation = 45)
        plt.xlim(0.3,len(to_show_ind)+1)
        plt.ylabel(data_name[j0])
        plt.grid()
    
    xtick = np.arange(1, n_pairs+1)-0.3
    for j0 in range(2):
        plt.subplot(2,2,j0*2+2)
        tmp_data = data_diff[j0]
        plt.bar(xtick,np.mean(tmp_data, axis = 0), width = width, color = 'r')
        plt.errorbar(xtick+width/2, np.mean(tmp_data, axis = 0), 
                     np.std(tmp_data,axis = 0)/np.sqrt(n_simu), fmt = None, lw = 2,ecolor = 'k')
        plt.xticks(xtick+width/2,xticknames_pair, rotation = 45)
        plt.xlim(0.3,n_pairs+1)
        plt.ylabel("diff " + data_name[j0])
        plt.grid()
    
    plt.tight_layout(0.2)
    
    plt.savefig("/home/ying/Dropbox/Thesis/Dissertation/Draft/Figures/Result_figures/roi_cov/roi_cov_L%d_summary.pdf" %L_list_option)
        
        
        
