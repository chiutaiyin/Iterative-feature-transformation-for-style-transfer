import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from utils.core import stylize_opt
from utils.core_seg import stylize_seg_opt
from utils.vgg_layerwise_SC import VggEnc
from utils import seg_utils

class DecoderBlock(layers.Layer):
    def __init__(self, block_num):
        super(DecoderBlock, self).__init__()
        assert block_num in [0,1,2,3], "Wrong block number!"
        self.block_num = block_num
        channel_dict = {3: 512, 2: 256, 1: 128, 0: 3}
        block_dict = {3: ['conv', 'upsampling', 'conv', 'conv', 'conv'],
                      2: ['conv', 'upsampling', 'conv'],
                      1: ['conv', 'upsampling', 'conv'],
                      0: ['conv']}
        
        self.layers = []
        c = channel_dict[block_num]
        if block_num != 0:
            for l in block_dict[block_num]:
                if l == 'conv':
                    self.layers.append(layers.Conv2D(c, (3,3), padding='same', activation='relu'))
                elif l == 'upsampling':
                    c = c // 2
                    self.layers.append(layers.UpSampling2D(interpolation='bilinear'))
        else:
            self.layers.append(layers.Conv2D(c, (3,3), padding='same', activation='linear'))
    
    def call(self, x, skip=None):
        x = self.layers[0](x)
        if self.block_num == 0:
            return x
        
        x = self.layers[1](x)
        x = x + tf.concat([skip, skip], -1) 
        for l in self.layers[2:]:
            x = l(x)
        return x
    
    
class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec = [DecoderBlock(i) for i in reversed(range(4))]
            
    def call(self, x, idx, skip=None):
        x = self.dec[idx](x, skip)
        return x
        
    def call_all(self, x41, skips):
        x31 = self.dec[0](x41, skips[0])
        x21 = self.dec[1](x31, skips[1])    
        x11 = self.dec[2](x21, skips[2])   
        x = self.dec[3](x11)    

        return x31, x21, x11, x
    
    
def ins_norm(feat):
    m_c = tf.reduce_mean(feat, axis=[1,2], keepdims=True)
    feat = feat - m_c
    s_c = tf.sqrt(tf.reduce_mean(feat * feat, axis=[1,2], keepdims=True) + 1e-8)
    feat = feat / s_c
    return feat
    
    
class EncDec(Model):
    def __init__(self):
        super(EncDec, self).__init__()        
        self.encoder = VggEnc()
        self.decoder = Decoder()
        self.iter_kargs = {
                            0: {'lr':0.01, 'lamb':1e4, 'n_iter':20},
                            1: {'lr':0.01, 'lamb':1e3, 'n_iter':20},
                            2: {'lr':0.01, 'lamb':1e2, 'n_iter':20},
                            3: {'lr':0.01, 'lamb':1e2, 'n_iter':20}
                          }
        
    def stylize_core(self, c_feat, s_feats, weights, opt, alpha, lr=0.01, lamb=1e2, n_iter=20):
        n_batch, cont_h, cont_w, n_channel = c_feat.shape
        _c_feat = tf.reshape(tf.transpose(c_feat, [0, 3, 1, 2]), [n_batch, n_channel, -1])
        if opt in ['iter']:
            c_feat = stylize_opt[opt](_c_feat, s_feats, weights, lr, lamb, n_iter) 
        else:
            c_feat = stylize_opt[opt](_c_feat, s_feats, weights)
        if opt in ['zca', 'ot', 'adain']:
            c_feat = (1 - alpha) * _c_feat + alpha * c_feat
        c_feat = tf.transpose(tf.reshape(c_feat, [n_batch, n_channel, cont_h, cont_w]), [0, 2, 3, 1])
        return c_feat

    def stylize(self, content, styles, enc_layers={3:'zca'}, dec_layers={0:'zca', 1:'zca', 2:'zca'},
                alpha=0.8, weights=None, iter_kargs=None):
        if weights is None:
            weights = [1 / len(styles) for _ in range(len(styles))]
            
        if not iter_kargs:
            iter_kargs = self.iter_kargs
            
        s_enc_feats = []  
        for style in styles:
            feats, skip_feats = self.encoder.call_all(style)
            s_enc_feats.append([tf.reshape(tf.transpose(feat, [0, 3, 1, 2]), [feat.shape[0], feat.shape[-1], -1])
                                for feat in feats])
          
        skips = [None]
        c_feat = self.encoder(content, 0)[0]
        if 0 in enc_layers.keys(): 
            c_feat = self.stylize_core(c_feat, [s_enc_feat[0] for s_enc_feat in s_enc_feats], weights,  
                                       enc_layers[0], alpha, **iter_kargs[0])

        for i in range(1,4):
            c_feat, skip_feat = self.encoder(c_feat, i)

            skips.append(ins_norm(skip_feat))
            if i in enc_layers.keys():
                c_feat = self.stylize_core(c_feat, [s_enc_feat[i] for s_enc_feat in s_enc_feats], weights,
                                           enc_layers[i], alpha, **iter_kargs[i])
    
        for i in range(4):
            c_feat = self.decoder(c_feat, i, skips[3-i])
            if 2-i in dec_layers.keys():
                c_feat = self.stylize_core(c_feat, [s_enc_feat[2-i] for s_enc_feat in s_enc_feats], weights,
                                           dec_layers[2-i], alpha, **iter_kargs[2-i])

        return c_feat
    
    def stylize_seg_core(self, c_feat, s_feat, c_seg_label_idx, s_seg_label_idx,
                         opt, alpha, lr=0.01, lamb=1e2, n_iter=20):
        _, cont_h, cont_w, n_channel = c_feat.shape
        _c_feat = tf.reshape(c_feat[0], [-1, n_channel])
        if opt in ['iter']:
            c_feat = stylize_seg_opt[opt](_c_feat, s_feat, c_seg_label_idx, s_seg_label_idx, lr, lamb, n_iter)
        else:
            c_feat = stylize_seg_opt[opt](_c_feat, s_feat, c_seg_label_idx, s_seg_label_idx)
        if opt in ['zca', 'ot', 'adain']:
            c_feat = (1 - alpha) * _c_feat + alpha * c_feat
        c_feat = tf.reshape(c_feat, [1, cont_h, cont_w, n_channel])
        return c_feat

    def stylize_seg(self, content, style, cont_seg, style_seg, enc_layers={3:'zca'}, dec_layers={0:'zca', 1:'zca', 2:'zca'},
                    alpha=0.8, iter_kargs=None):            
        if not iter_kargs:
            iter_kargs = self.iter_kargs
            
        label_set, label_ind = seg_utils.compute_label_info(cont_seg, style_seg)
        cont_segs = [cont_seg]
        style_segs = [style_seg]
        for i in range(1,4):
            cont_segs.append(cv2.resize(cont_seg, 
                                        dsize=(np.ceil(cont_segs[i-1].shape[1] / 2).astype(np.int32), 
                                               np.ceil(cont_segs[i-1].shape[0] / 2).astype(np.int32)), 
                                        interpolation=cv2.INTER_NEAREST))
            style_segs.append(cv2.resize(style_seg, 
                                         dsize=(np.ceil(style_segs[i-1].shape[1] / 2).astype(np.int32), 
                                                np.ceil(style_segs[i-1].shape[0] / 2).astype(np.int32)), 
                                         interpolation=cv2.INTER_NEAREST))
        cont_seg_label_idx = [[np.reshape(cont_segs[i], (-1,)) == l for l in label_set if label_ind[l]] for i in range(4)]
        style_seg_label_idx = [[np.reshape(style_segs[i], (-1,)) == l for l in label_set if label_ind[l]] for i in range(4)]

        
        feats, skip_feats = self.encoder.call_all(style)
        s_enc_feat = [tf.reshape(feat[0], [-1, feat.shape[-1]]) for feat in feats]
        s_skip_feat = [tf.reshape(feat[0], [-1, feat.shape[-1]]) for feat in skip_feats[::-1]]                            
          
        skips = [None]
        c_feat = self.encoder(content, 0)[0]
        if 0 in enc_layers.keys(): 
            c_feat = self.stylize_seg_core(c_feat, s_enc_feat[0], cont_seg_label_idx[0], style_seg_label_idx[0],  
                                       enc_layers[0], alpha, **iter_kargs[0])
        for i in range(1,4):
            c_feat, skip_feat = self.encoder(c_feat, i)

            skips.append(ins_norm(skip_feat))
            if i in enc_layers.keys():
                c_feat = self.stylize_seg_core(c_feat, s_enc_feat[i], cont_seg_label_idx[i], style_seg_label_idx[i],
                                           enc_layers[i], alpha, **iter_kargs[i])
                
        for i in range(4):
            c_feat = self.decoder(c_feat, i, skips[3-i])
            if 2-i in dec_layers.keys():
                c_feat = self.stylize_seg_core(c_feat, s_enc_feat[2-i], cont_seg_label_idx[2-i], style_seg_label_idx[2-i],
                                           dec_layers[2-i], alpha, **iter_kargs[2-i])

        return c_feat
    
            
    def call(self, content):        
        x, y = self.encoder.call_all(content)
        x = self.decoder.call_all(x[-1], y)
        return x           
    
    
    
    