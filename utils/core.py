import tensorflow as tf
import numpy as np
from utils.aux import *

# shape of feat = (batch_size, C, HW)

def stylize_zca(c_feat, s_feats, weights): 
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_ss = [tf.reduce_mean(s_feat, axis=-1, keepdims=True) for s_feat in s_feats]
    c_feat = c_feat - m_c
    s_feats = [s_feat - m_s for s_feat, m_s in zip(s_feats, m_ss)]
    c_cov = tf.matmul(c_feat, c_feat, transpose_b=True) / c_feat.shape[-1]
    s_covs = [tf.matmul(s_feat, s_feat, transpose_b=True) / s_feat.shape[-1] for s_feat in s_feats]
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
    opts = [tf.matmul(inv_sqrt_cov(s_cov), inv_sqrt_c_cov) for s_cov in s_covs]
    feats = [tf.matmul(opt, c_feat) + m_s for opt, m_s in zip(opts, m_ss)]
    feat = tf.reduce_sum(tf.multiply(feats, tf.reshape(weights, [len(weights), 1, 1, 1])), axis=0)
    return feat


def stylize_ot(c_feat, s_feats, weights):
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_ss = [tf.reduce_mean(s_feat, axis=-1, keepdims=True) for s_feat in s_feats]
    c_feat = c_feat - m_c
    s_feats = [s_feat - m_s for s_feat, m_s in zip(s_feats, m_ss)]
    c_cov = tf.matmul(c_feat, c_feat, transpose_b=True) / c_feat.shape[-1]
    s_covs = [tf.matmul(s_feat, s_feat, transpose_b=True) / s_feat.shape[-1] for s_feat in s_feats]
    sqrt_c_cov = inv_sqrt_cov(c_cov)
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
    opts = [inv_sqrt_cov(tf.matmul(sqrt_c_cov, tf.matmul(s_cov, sqrt_c_cov))) for s_cov in s_covs]
    opts = [tf.matmul(inv_sqrt_c_cov, tf.matmul(opt, inv_sqrt_c_cov)) for opt in opts]
    feats = [tf.matmul(opt, c_feat) + m_s for opt, m_s in zip(opts, m_ss)]
    feat = tf.reduce_sum(tf.multiply(feats, tf.reshape(weights, [len(weights), 1, 1, 1])), axis=0)
    return feat

    
def stylize_adain(c_feat, s_feats, weights):
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_ss = [tf.reduce_mean(s_feat, axis=-1, keepdims=True) for s_feat in s_feats]
    c_feat = c_feat - m_c
    s_feats = [s_feat - m_s for s_feat, m_s in zip(s_feats, m_ss)]
    s_c = tf.sqrt(tf.reduce_mean(c_feat * c_feat, axis=-1, keepdims=True) + 1e-8)
    s_ss = [tf.sqrt(tf.reduce_mean(s_feat * s_feat, axis=-1, keepdims=True) + 1e-8) for s_feat in s_feats]
    white_c_feat = c_feat / s_c
    feats = [white_c_feat * s_s + m_s for s_s, m_s in zip(s_ss, m_ss)]
    feat = tf.reduce_sum(tf.multiply(feats, tf.reshape(weights, [len(weights), 1, 1, 1])), axis=0)
    return feat


def stylize_iter(c_feat, s_feats, weights, lr=0.01, lamb=1e2, n_iter=20):
    sum_w = sum(weights)
    lamb = lamb * sum_w
    FsFsT = [tf.matmul(style_feat, style_feat, transpose_b=True) / style_feat.shape[-1] * weights[i] / sum_w \
             for i, style_feat in enumerate(s_feats)]
    sum_FsFsT = sum(FsFsT)
    
    feat = c_feat
    n = c_feat.shape[-1]
    for _ in range(n_iter):
        grad = 2 * (feat - c_feat) + 4 * lamb / n * tf.matmul(
               tf.matmul(feat, feat, transpose_b=True) / n - sum_FsFsT, feat)
        feat = feat - lr * grad
    return feat

stylize_opt = {'zca': stylize_zca, 'ot': stylize_ot, 'adain': stylize_adain, 'iter': stylize_iter}