import tensorflow as tf
import numpy as np

def inv_sqrt_cov(cov, inverse=False):
    s, u, _ = tf.linalg.svd(cov + tf.eye(cov.shape[-1]))
    s = tf.sqrt(s)
    if inverse:
        s = 1 / s
    d = tf.linalg.diag(s)
    return tf.matmul(u, tf.matmul(d, u, adjoint_b=True))

def cubic_solver(a, b, c, d): # shape of a, b, c, d = (N,1)
    a = tf.complex(a, 0.0)
    b = tf.complex(b, 0.0)
    c = tf.complex(c, 0.0)
    d = tf.complex(d, 0.0)
    delta_0 = b*b - 3*a*c
    delta_1 = 2*b*b*b - 9*a*b*c + 27*a*a*d
    inner_sqrt = tf.sqrt(delta_1 * delta_1 - 4 * delta_0 * delta_0 * delta_0)
    C = tf.concat([delta_1+inner_sqrt, delta_1-inner_sqrt], axis=-1)
    idx = tf.stack([tf.range(C.shape[0]),
                    tf.argmax(tf.abs(C), -1, tf.int32)], axis=-1)
    C = tf.pow(tf.expand_dims(tf.gather_nd(C, idx), -1) / 2, 1/3)
    Xi = [[1.0, tf.complex(-0.5, tf.sqrt(3.0)/2), tf.complex(-0.5, -tf.sqrt(3.0)/2)]]
    C = C * Xi # shape = (N, 3)
    x = -(b + C + delta_0 / (C + 1e-8)) / (3 * a)
    return x
    
def eta_selection(x):
    idx = tf.where(tf.abs(tf.math.imag(x)) > 1e-5)
    x = tf.tensor_scatter_nd_update(tf.math.real(x), idx, tf.zeros(idx.shape[0]))
    x = tf.reduce_max(x, axis=-1, keepdims=True) # shape = (N, 1)
    return x