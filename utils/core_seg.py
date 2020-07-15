import tensorflow as tf
import numpy as np
from utils.aux import *

# shape of feat = (HW, C)

def stylize_zca(c_feat, s_feat, c_seg_label_idx, s_seg_label_idx):
    feat = c_feat
    for i in range(len(c_seg_label_idx)):
        cl = c_seg_label_idx[i]
        sl = s_seg_label_idx[i]
        f = tf.boolean_mask(c_feat, cl, axis=0)
        sf = tf.boolean_mask(s_feat, sl, axis=0)
    
        m_c = tf.reduce_mean(f, axis=0, keepdims=True)
        m_s = tf.reduce_mean(sf, axis=0, keepdims=True)
        f = f - m_c
        sf = sf - m_s 
        c_cov = tf.matmul(f, f, transpose_a=True) / f.shape[0]
        s_cov = tf.matmul(sf, sf, transpose_a=True) / sf.shape[0] 
        inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
        opt = tf.matmul(inv_sqrt_c_cov, inv_sqrt_cov(s_cov)) 
        f = tf.matmul(f, opt) + m_s 
        
        cl = tf.where(cl)
        feat = tf.tensor_scatter_nd_update(feat, cl, f) 
    return feat


def stylize_ot(c_feat, s_feat, c_seg_label_idx, s_seg_label_idx):
    feat = c_feat
    for i in range(len(c_seg_label_idx)):
        cl = c_seg_label_idx[i]
        sl = s_seg_label_idx[i]
        f = tf.boolean_mask(c_feat, cl, axis=0)
        sf = tf.boolean_mask(s_feat, sl, axis=0)
        
        m_c = tf.reduce_mean(f, axis=0, keepdims=True)
        m_s = tf.reduce_mean(sf, axis=0, keepdims=True) 
        f = f - m_c
        sf = sf - m_s 
        c_cov = tf.matmul(f, f, transpose_a=True) / f.shape[0]
        s_cov = tf.matmul(sf, sf, transpose_a=True) / sf.shape[0] 
        sqrt_c_cov = inv_sqrt_cov(c_cov)
        inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
        opt = inv_sqrt_cov(tf.matmul(sqrt_c_cov, tf.matmul(s_cov, sqrt_c_cov)))
        opt = tf.matmul(inv_sqrt_c_cov, tf.matmul(opt, inv_sqrt_c_cov)) 
        f = tf.matmul(f, opt) + m_s 
        
        cl = tf.where(cl)
        feat = tf.tensor_scatter_nd_update(feat, cl, f) 
    return feat

    
def stylize_adain(c_feat, s_feat, c_seg_label_idx, s_seg_label_idx):
    feat = c_feat
    for i in range(len(c_seg_label_idx)):
        cl = c_seg_label_idx[i]
        sl = s_seg_label_idx[i]
        f = tf.boolean_mask(c_feat, cl, axis=0)
        sf = tf.boolean_mask(s_feat, sl, axis=0)
        
        m_c = tf.reduce_mean(f, axis=0, keepdims=True)
        m_s = tf.reduce_mean(sf, axis=0, keepdims=True)
        f = f - m_c
        sf = sf - m_s 
        s_c = tf.sqrt(tf.reduce_mean(f * f, axis=0, keepdims=True) + 1e-8)
        s_s = tf.sqrt(tf.reduce_mean(sf * sf, axis=0, keepdims=True) + 1e-8) 
        white_c_feat = f / s_c
        f = white_c_feat * s_s + m_s 
        
        cl = tf.where(cl)
        feat = tf.tensor_scatter_nd_update(feat, cl, f) 
    return feat


def stylize_iter(c_feat, s_feat, c_seg_label_idx, s_seg_label_idx, lr=0.01, lamb=1e2, n_iter=20):
    FsFsT = []
    for i in range(len(c_seg_label_idx)):
        sl = s_seg_label_idx[i]
        sf = tf.boolean_mask(s_feat, sl, axis=0)
        FsFsT.append(tf.matmul(sf, sf, transpose_a=True) / sf.shape[0])

    feat = c_feat
    for _ in range(n_iter):
        grad = 2 * (feat - c_feat)
        for i in range(len(c_seg_label_idx)):
            cl = c_seg_label_idx[i]            
            f = tf.boolean_mask(feat, cl, axis=0)
            FsFsT_ = FsFsT[i]
            grad_ = tf.boolean_mask(grad, cl, axis=0)
            upd = 4 * lamb / feat.shape[0] * tf.matmul(f, (tf.matmul(f, f, transpose_a=True) / f.shape[0] - FsFsT_))
            
            cl = tf.where(cl)
            feat = tf.tensor_scatter_nd_update(feat, cl, f - lr * (grad_ + upd))        
    return feat

###= ==================================================== under test ========================================= ###
def whiten_zca(c_feat): 
    m_c = tf.reduce_mean(c_feat, axis=0, keepdims=True)
    c_feat = c_feat - m_c    
    c_cov = tf.matmul(c_feat, c_feat, transpose_a=True) / c_feat.shape[0]    
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
    f = tf.matmul(c_feat, inv_sqrt_c_cov) + m_c

    return f

def whiten_in(c_feat): 
    m_c = tf.reduce_mean(c_feat, axis=0, keepdims=True)
    c_feat = c_feat - m_c    
    s_c = tf.sqrt(tf.reduce_mean(c_feat * c_feat, axis=0, keepdims=True) + 1e-8)       
    f = c_feat / s_c + m_c

    return f

def whiten_line(c_feat, lamb=1e4, n_iter=3):      
    FsFsT = tf.eye(c_feat.shape[-1])
    
    feat = c_feat
    n = c_feat.shape[0]        
    for _ in range(n_iter):
        cov_diff = tf.matmul(feat, feat, transpose_a=True) / n - FsFsT
        grad = 2 * (feat - c_feat) + 4 * lamb / n * tf.matmul(feat, cov_diff)
        DD_T = tf.matmul(grad, grad, transpose_a=True)
        DF_T = tf.matmul(grad, feat, transpose_a=True)
        a = 2 * lamb / (n * n) * tf.reduce_mean(DD_T * DD_T, keepdims=True)
        b = -6 * lamb / (n * n) * tf.reduce_mean(DF_T * DD_T, keepdims=True)
        tmp1 = tf.linalg.trace(DD_T) / c_feat.shape[1] / c_feat.shape[1]
        tmp2 = 2 * lamb / (n * n) * (tf.reduce_mean(DF_T * DF_T) + 
                                     tf.reduce_mean(DF_T * tf.transpose(DF_T)))
        tmp3 = 2 * lamb / n * tf.reduce_mean(DD_T * cov_diff)
        c = tf.reshape(tmp1 + tmp2 + tmp3, [1,1]) 
        d = tf.reshape(-0.5 * tmp1, [1,1]) 
        # scaling to avoid overflow
        abcd = tf.concat([a,b,c,d], axis=-1)
        scale = tf.reduce_max(abcd, axis=-1, keepdims=True)
        abcd = abcd / scale
        eta = eta_selection(cubic_solver(abcd[:,0:1], abcd[:,1:2], abcd[:,2:3], abcd[:,3:]))
        feat = feat - eta * grad
    return feat

###= ==================================================== under test ========================================= ###


def stylize_line(c_feat, s_feat, c_seg_label_idx, s_seg_label_idx, lr=None, lamb=1e2, n_iter=1):
    FsFsT = []
    for i in range(len(c_seg_label_idx)):
        sl = s_seg_label_idx[i]
        sf = tf.boolean_mask(s_feat, sl, axis=0)
        FsFsT.append(tf.matmul(sf, sf, transpose_a=True) / sf.shape[0])
    
    feat = c_feat       
    n = c_feat.shape[0]        
    for _ in range(n_iter):
        grad_all = tf.zeros_like(feat)
        a, b, c, d = 0, 0, 0, 0
        for i in range(len(c_seg_label_idx)):
            cl = c_seg_label_idx[i]     
            cf = tf.boolean_mask(c_feat, cl, axis=0)
            f = tf.boolean_mask(feat, cl, axis=0)
            cov_diff = tf.matmul(f, f, transpose_a=True) / f.shape[0] - FsFsT[i]
            grad = 2 * (f - cf) + 4 * lamb / n * tf.matmul(f, cov_diff) 
            grad_all = tf.tensor_scatter_nd_update(grad_all, tf.where(cl), grad)
            DD_T = tf.matmul(grad, grad, transpose_a=True)
            DF_T = tf.matmul(grad, f, transpose_a=True)
            a += 2 * lamb / (n * f.shape[0]) * tf.reduce_mean(DD_T * DD_T, keepdims=True)
            b += -6 * lamb / (n * f.shape[0]) * tf.reduce_mean(DF_T * DD_T, keepdims=True)
            tmp1 = tf.linalg.trace(DD_T) / c_feat.shape[1] / c_feat.shape[1]
            tmp2 = 2 * lamb / (n * f.shape[0]) * (tf.reduce_mean(DF_T * DF_T) + 
                                         tf.reduce_mean(DF_T * tf.transpose(DF_T)))
            tmp3 = 2 * lamb / n * tf.reduce_mean(DD_T * cov_diff)
            c += tf.reshape(tmp1 + tmp2 + tmp3, [1,1]) 
            d += tf.reshape(-0.5 * tmp1, [1,1]) 
        # scaling to avoid overflow
        abcd = tf.concat([a,b,c,d], axis=-1)
        scale = tf.reduce_max(abcd, axis=-1, keepdims=True)
        abcd = abcd / scale
        eta = eta_selection(cubic_solver(abcd[:,0:1], abcd[:,1:2], abcd[:,2:3], abcd[:,3:]))
        feat = feat - eta * grad_all
#         print([abcd[:,0:1], abcd[:,1:2], abcd[:,2:3], abcd[:,3:]])
#         print(eta)
    return feat

stylize_seg_opt = {'zca': stylize_zca, 'ot': stylize_ot, 'adain': stylize_adain, 'iter': stylize_iter, 'line': stylize_line}