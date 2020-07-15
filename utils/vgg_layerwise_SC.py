import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, UpSampling2D, Activation, Lambda, MaxPooling2D
import torchfile

def pad_reflect(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')

def vgg_from_t7(t7_file='./utils/vgg_normalised.t7',
                skip_layers=['relu1_2', 'relu2_2', 'relu3_4'],
                target_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']):
    '''Extract VGG layers from a Torch .t7 model into a Keras model
       e.g. vgg = vgg_from_t7('vgg_normalised.t7', target_layer='relu4_1')
       Adapted from https://github.com/jonrei/tf-AdaIN/blob/master/AdaIN.py
       Converted caffe->t7 from https://github.com/xunhuang1995/AdaIN-style
    '''
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    encs = {0:[[]], 1:[[], []], 2:[[], []], 3:[[], []]}
    
    for idx,module in enumerate(t7.modules):
        name = module.name.decode() if module.name is not None else None
        
        if idx == 0:
            name = 'preprocess'  # VGG 1st layer preprocesses with a 1x1 conv to multiply by 255 and subtract BGR mean as bias
            i = 0
            j = 0

        if module._typename == b'nn.SpatialReflectionPadding':
            encs[i][j].append(Lambda(pad_reflect))
        elif module._typename == b'nn.SpatialConvolution':
            filters = module.nOutputPlane
            kernel_size = module.kH
            weight = module.weight.transpose([2,3,1,0]).astype(np.float32)
            bias = module.bias.astype(np.float32)
            encs[i][j].append(Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=tf.constant_initializer(weight),
                        bias_initializer=tf.constant_initializer(bias),
                        trainable=False))
        elif module._typename == b'nn.ReLU':
            encs[i][j].append(Activation('relu', name=name))
            if name in skip_layers:
                j += 1
                
            if name in target_layers:
                i += 1
                j = 0
        elif module._typename == b'nn.SpatialMaxPooling':
            encs[i][j].append(MaxPooling2D(padding='same', name=name))            
        # elif module._typename == b'nn.SpatialUpSamplingNearest': # Not needed for VGG
        #     x = Upsampling2D(name=name)(x)
        else:
            raise NotImplementedError(module._typename)

        if name == target_layers[-1]:
            # print("Reached target layer", target_layer)
            break
    
    return encs

vgg_encs = vgg_from_t7()

class VggEncBlock(Layer):
    def __init__(self, idx):
        super(VggEncBlock, self).__init__()
        self.layers = vgg_encs[idx]
    
    def call(self, x):
        outputs = []
        for layers in self.layers:
            for l in layers:
                x = l(x)
            outputs.append(x)
        return outputs[::-1] # outputs[0] for forward passing, outputs[1] for skip connection if any

class VggEnc(Layer):
    def __init__(self):
        super(VggEnc, self).__init__()
        self.enc = [VggEncBlock(idx) for idx in range(4)]
        
    def call(self, x, idx):
        x = self.enc[idx](x)
        return x # x is a list
    
    def call_all(self, x):
        x11 = self.enc[0](x)
        x21 = self.enc[1](x11[0])
        x31 = self.enc[2](x21[0])
        x41 = self.enc[3](x31[0])
        return (x11[0], x21[0], x31[0], x41[0]), (x41[1], x31[1], x21[1])