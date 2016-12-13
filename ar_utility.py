# -*- coding: utf-8 -*-
import numpy as np
import sys
import scipy.io

import matplotlib.pyplot as plt

# no depth weithing is reapplied to the source solution

# for portability, I need to make write it as a package
path = "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/"
sys.path.insert(0, path)
from ROI_Kalman_smoothing import (get_param_given_u, simulate_kalman_filter_data)


# directly estimate Ath for u_t+h = Ath u_t + error
# with ordinary least square retression
def least_square_lagged_regression(u_array):
    """
    u_array, q, T+1, p
    """
    q,T,p = u_array.shape
    T -= 1
    # t0, t1 term is t1 regressed on t0
    lagged_coef_mat = np.zeros([T,T],dtype = np.object)
    for t0 in range(T):
        for t1 in range(t0,T):
            tmp_coef = np.zeros([p,p])
            for i in range(p):
                # least square regression u_t+h[i]  u_t
                tmp_y = u_array[:,t1+1,i]
                tmp_x = u_array[:,t0,:]
                # (X'X)^{-1} X' Y
                tmp_coef[i,:] = np.linalg.inv(tmp_x.T.dot(tmp_x)).dot(tmp_x.T.dot(tmp_y))
                
            lagged_coef_mat[t0,t1] = tmp_coef
                
    return lagged_coef_mat

def get_tilde_A_from_At(A):
    """
        A[T,p,p]
    """
    T, p, _ = A.shape
    tilde_A = np.zeros([T,T], dtype = np.object)  
    for i0 in range(T):
        tilde_A[i0,i0] = A[i0].copy()
    for i0 in range(T):
        for j0 in range(i0+1,T):
            tmp = np.eye(p)
            for l0 in range(j0,i0-1,-1):
                tmp = (np.dot(tmp, A[l0])).copy()
            tilde_A[i0,j0] = tmp
    return tilde_A
    
def get_dynamic_linear_coef(tildeA, pair):
    T = tildeA.shape[0]
    tilde_A_entry = np.zeros([T,T])
    for t1 in range(T):
        for t2 in range(T):
            if t1<= t2:
                tilde_A_entry[t1,t2] = tildeA[t1,t2][pair[1], pair[0]]
            else:
                # region 2 -> region 1 lower diangonal feedback
                tilde_A_entry[t1,t2] = tildeA[t2,t1][pair[0], pair[1]] 
    
    return tilde_A_entry
#============================    
    


