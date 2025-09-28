import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import numpy as np
        

 
def prelu(x):
    with tf.name_scope('PRELU'):
        _alpha = tf.get_variable('prelu', shape=x.get_shape()[-1], dtype = x.dtype, initializer=tf.constant_initializer(0.0))
    return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)



def conv2d(input, name, n_channels_output=None, kernel_size=3, strides=[1,1,1,1], padding='SAME', add_bias=True, act='leakyrelu'):
    with tf.variable_scope(name):
        n_channels_input = input.get_shape()[-1].value
        kernel_shape = [kernel_size, kernel_size, n_channels_input, n_channels_output]
    #    weights = tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.get_variable('weights', shape=kernel_shape, initializer=tf.glorot_uniform_initializer())
        output = tf.nn.conv2d(input, weights, strides=strides, padding=padding)
        
        if add_bias:
            biases = tf.get_variable('biases', shape=[n_channels_output], initializer=tf.constant_initializer(0.1))
            output = tf.nn.bias_add(output, biases)
        
        if act == 'prelu': output = prelu(output)
        elif act == 'relu': output = tf.nn.relu(output)
        elif act == 'tanh': output = tf.nn.tanh(output)
        elif act == 'sigmoid': output = tf.sigmoid(output)
        elif act == 'leakyrelu': output = tf.nn.leaky_relu(output)
        elif act == None: pass
    return output      
      
    
    
def pool2d(input, kernel_size, stride, name, padding='SAME', use_avg=True):
    with tf.variable_scope(name):
        if use_avg: 
            return tf.nn.avg_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
        else: 
            return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
        


def fully_connected(input, n_nodes_output, name, add_bias=True, act='relu'):           
    with tf.variable_scope(name):        
        n_nodes_input = input.get_shape()[-1].value
        weights_shape = [n_nodes_input, n_nodes_output]
    #    weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.glorot_uniform_initializer())        
        output = tf.matmul(input, weights)
        
        if add_bias:
            biases = tf.get_variable('biases', shape=[n_nodes_output], initializer=tf.constant_initializer(0.1))
            output = tf.nn.bias_add(output, biases)

        if act == 'prelu': output = prelu(output)
        elif act == 'relu': output = tf.nn.relu(output)
        elif act == 'tanh': output = tf.nn.tanh(output)
        elif act == 'sigmoid': output = tf.sigmoid(output)
        elif act == 'leakyrelu': output = tf.nn.leaky_relu(output)
        elif act == None: pass
        return output



def inception_Treyer(input, kernels, name, act='relu'):
    with tf.variable_scope(name):
        a0 = conv2d(input=input, n_channels_output=int(kernels*0.65), kernel_size=1, name='a0', act=act)
        a1 = conv2d(input=a0, n_channels_output=kernels, kernel_size=5, name='a1', act=act)
        b0 = conv2d(input=input, n_channels_output=int(kernels*0.65), kernel_size=1, name='b0', act=act)
        b1 = conv2d(input=b0, n_channels_output=kernels, kernel_size=3, name='b1', act=act)
        c0 = conv2d(input=input, n_channels_output=int(kernels*0.65), kernel_size=1, name='c0', act=act)
        c1 = pool2d(input=c0, kernel_size=2, name='c1', stride=1, use_avg=True)
        d1 = conv2d(input=input, n_channels_output=int(kernels*0.7), kernel_size=1, name='d1', act=act)
    return tf.concat([a1, b1, c1, d1], 3)



class Model:
    def __init__(self, texp, config, dim_latent_main, dim_latent_ext, img_size, c_input, c_inputadd, bins, name):
        self.texp = texp
        self.config = config
        self.dim_latent_main = dim_latent_main
        self.dim_latent_ext = dim_latent_ext
        self.c_input = c_input
        self.bins = bins
        self.name = name
                        
        if config == 0:  # photometry-only 
            self.x = tf.placeholder(tf.float32, shape=[None, c_input], name='x')
            self.x2 = tf.placeholder(tf.float32, shape=[None, c_input], name='x2')
        elif config == 1:  # image-based
            self.x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, c_input], name='x')
            self.x2 = tf.placeholder(tf.float32, shape=[None, img_size, img_size, c_input], name='x2')
            self.x_morph = tf.placeholder(tf.float32, shape=[None, img_size, img_size, c_input], name='x_morph')
            self.x2_morph = tf.placeholder(tf.float32, shape=[None, img_size, img_size, c_input], name='x2_morph')

        self.inputadd = tf.placeholder(tf.float32, shape=[None, c_inputadd], name='inputadd')
        self.inputadd2 = tf.placeholder(tf.float32, shape=[None, c_inputadd], name='inputadd2')

        self.y = tf.placeholder(tf.float32, shape=[None, bins], name='y')
        self.y2 = tf.placeholder(tf.float32, shape=[None, bins], name='y2')

        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        

            
    def encoder(self, x, inputadd, name, reuse):
        with tf.variable_scope(self.name + name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                
            if self.config == 0:  # photometry-only
                act = 'leakyrelu'
                x_concat = tf.concat([x, inputadd], 1)
                fc0 = x_concat
                for i in range(20):
                    fc0 = fully_connected(input=fc0, n_nodes_output=128, name='fc0_'+str(i), act=act)
                fc0 = fully_connected(input=fc0, n_nodes_output=self.dim_latent_main + self.dim_latent_ext, name='fc0_f', act=None)                    

            elif self.config == 1:  # image-based; InceptionNet_Treyer
                act = 'relu'
                fc_add = fully_connected(input=inputadd, n_nodes_output=96, name='fc_add', act=None) 
                conv1 = conv2d(input=x, n_channels_output=96, kernel_size=5, name='conv1', act=None)                    
                conv1e = tf.expand_dims(tf.expand_dims(fc_add, 1), 1)
                conv1 = tf.nn.relu(conv1 + conv1e)    
                    
                conv2 = conv2d(input=conv1, n_channels_output=96, kernel_size=3, name='conv2', act='tanh')
                pool2 = pool2d(input=conv2, kernel_size=2, stride=2, name='pool2', use_avg=True)
                inc1 = inception_Treyer(input=pool2, kernels=156, name='inc1', act=act)
                inc2 = inception_Treyer(input=inc1, kernels=156, name='inc2', act=act)
                inc2b = inception_Treyer(input=inc2, kernels=156, name='inc2b', act=act)
                pool3 = pool2d(input=inc2b, kernel_size=2, stride=2, name='pool3', use_avg=True)
                inc3 = inception_Treyer(input=pool3, kernels=156, name='inc3', act=act)
                inc3b = inception_Treyer(input=inc3, kernels=156, name='inc3b', act=act)
                pool4 = pool2d(input=inc3b, kernel_size=2, stride=2, name='pool4', use_avg=True)
                inc4 = inception_Treyer(input=pool4, kernels=156, name='inc4', act=act)
                conv5 = conv2d(input=inc4, n_channels_output=96, kernel_size=3, name='conv5', act=act, padding='VALID')
                conv6 = conv2d(input=conv5, n_channels_output=96, kernel_size=3, name='conv6', act=act, padding='VALID')
                conv7 = conv2d(input=conv6, n_channels_output=96, kernel_size=3, name='conv7', act=act, padding='VALID')
                dede = pool2d(input=conv7, kernel_size=2, stride=1, name='dede', use_avg=True)
                flat = tf.layers.Flatten()(dede)
             #   print (flat)                        
                fc0 = fully_connected(input=flat, n_nodes_output=1024, name='fc0', act=act)       
                fc0 = fully_connected(input=fc0, n_nodes_output=self.dim_latent_main + self.dim_latent_ext, name='fc0_f', act=None)
            return fc0            

                

    def decoder(self, latent, name, reuse):
        with tf.variable_scope(self.name + name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            
            if self.config == 0:  # photometry-only
                act = 'leakyrelu'
                fcd = latent
                for i in range(19):
                    fcd = fully_connected(input=fcd, n_nodes_output=128, name='fcd_'+str(i), act=act)
                fcd = fully_connected(input=fcd, n_nodes_output=self.c_input, name='fcd_f', act=None)
                return fcd
            
            elif self.config == 1:  # image-based
                act = 'leakyrelu'
                fc_de = fully_connected(input=latent, n_nodes_output=1024, name='fc_de', act=act)
                fm1 = tf.reshape(fc_de, [-1, 8, 8, 16])
                fm2 = conv2d(input=fm1, n_channels_output=32, kernel_size=3, name='fm2', act=act)
                fm3 = conv2d(input=fm2, n_channels_output=32, kernel_size=3, name='fm3', act=act)
                fm4 = tf.image.resize_images(fm3, size=[16, 16], method=tf.image.ResizeMethod.BILINEAR)
                fm5 = conv2d(input=fm4, n_channels_output=32, kernel_size=3, name='fm5', act=act)
                fm6 = conv2d(input=fm5, n_channels_output=32, kernel_size=3, name='fm6', act=act)
                fm7 = tf.image.resize_images(fm6, size=[32, 32], method=tf.image.ResizeMethod.BILINEAR)
                fm8 = conv2d(input=fm7, n_channels_output=32, kernel_size=3, name='fm8', act=act)
                fm9 = conv2d(input=fm8, n_channels_output=32, kernel_size=3, name='fm9', act=act)
                fm10 = tf.image.resize_images(fm9, size=[64, 64], method=tf.image.ResizeMethod.BILINEAR)
                fm11 = conv2d(input=fm10, n_channels_output=32, kernel_size=3, name='fm11', act=act)
                fm12 = conv2d(input=fm11, n_channels_output=self.c_input, kernel_size=3, name='fm12', act=None)
                return fm12
            
            
    
    def estimator(self, latent, name, reuse):        
        with tf.variable_scope(self.name + name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
       
            latent_input = latent[:, :self.dim_latent_main]
            if self.config == 0:  # photometry-only
                fc1 = fully_connected(input=latent_input, n_nodes_output=128, name='fc1', act='leakyrelu')
            elif self.config == 1:  # image-based
                fc1 = fully_connected(input=latent_input, n_nodes_output=1024, name='fc1', act='relu')
            fc2 = fully_connected(input=fc1, n_nodes_output=self.bins, name='fc2', act=None)
            return fc2
                
            

    def get_mse(self, x, x_recon):
        return tf.reduce_mean(tf.pow(x - x_recon, 2))

    
    
    def get_p_cross_entropy(self, y, ylogits):
        p = tf.nn.softmax(ylogits)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=ylogits))
        return p, cost


    
    def get_ylogits_contrast(self, ylogits1, ylogits2):
        p1 = tf.stop_gradient(tf.nn.softmax(ylogits1))
        p2 = tf.stop_gradient(tf.nn.softmax(ylogits2))
        cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p1, logits=ylogits2))
        cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p2, logits=ylogits1))
        return cost1 + cost2

    

    def get_outputs_single_pre(self, x, inputadd, use_2nd=False):
        latent = self.encoder(x, inputadd, name='encoder', reuse=use_2nd)
        x_recon = self.decoder(latent, name='decoder', reuse=use_2nd)
        ylogits = self.estimator(latent, name='estimator', reuse=use_2nd)
        return latent, x_recon, ylogits
        
    

    def get_outputs_single_cyc(self, x, y, inputadd, use_2nd=False):
        latent_cyc1, x_recon_cyc1, ylogits_cyc1 = self.get_outputs_single_pre(x, inputadd, use_2nd=use_2nd)
        latent_cyc2, x_recon_cyc2, ylogits_cyc2 = self.get_outputs_single_pre(x_recon_cyc1, inputadd, use_2nd=True)
            
        p1, cost_ce_single_cyc1 = self.get_p_cross_entropy(y, ylogits_cyc1)
        p2, cost_ce_single_cyc2 = self.get_p_cross_entropy(y, ylogits_cyc2)
            
        cost_recon_single_cyc1 = self.get_mse(x, x_recon_cyc1)
        cost_recon_single_cyc2 = self.get_mse(x, x_recon_cyc2)
  
        cost_ycontra = self.get_ylogits_contrast(ylogits_cyc1, ylogits_cyc2)       
            
        cost_ce = cost_ce_single_cyc1 + cost_ce_single_cyc2
        cost_recon = cost_recon_single_cyc1 + cost_recon_single_cyc2
            
        return cost_ce, cost_recon, cost_ycontra, p1, p2, ylogits_cyc1, cost_ce_single_cyc1, cost_ce_single_cyc2, latent_cyc1, latent_cyc2
            
                

    def get_outputs_aug(self, x_morph, inputadd, ylogits, use_2nd=False):
        latent_morphaug, x_recon_morphaug, ylogits_morphaug = self.get_outputs_single_pre(x_morph, inputadd, use_2nd=use_2nd)

        cost_recon_morphaug = self.get_mse(x_morph, x_recon_morphaug)
        cost_ycontra_morphaug = self.get_ylogits_contrast(ylogits, ylogits_morphaug)  
        return cost_recon_morphaug, cost_ycontra_morphaug, latent_morphaug
        


    def exp_minus_lmse(self, latent1_list, latent2_list):
        latent1_main = tf.concat(latent1_list, 0)[:, :self.dim_latent_main]
        latent2_main = tf.concat(latent2_list, 0)[:, :self.dim_latent_main]
        return tf.exp(-1 * tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.pow(latent1_main - latent2_main, 2), 1))))

    

    def get_outputs(self):
        if self.texp == 0:  # mutual information estimation
            latent = self.encoder(self.x, self.inputadd, name='encoder', reuse=False)
            ylogits = self.estimator(latent, name='estimator', reuse=False)
            p, cost_ce_single = self.get_p_cross_entropy(self.y, ylogits)
            return [p], [cost_ce_single], [latent[:, :self.dim_latent_main]]
                    
        elif self.texp == 1:  # supervised contrastive learning for causal analysis
            cost_ce1, cost_recon1, cost_ycontra1, p11, p12, ylogits1, cost_ce_single11, cost_ce_single12, latent1, latent1_ = self.get_outputs_single_cyc(self.x, self.y, self.inputadd, use_2nd=False)
            cost_ce2, cost_recon2, cost_ycontra2, _, _, ylogits2, _, _, latent2, latent2_ = self.get_outputs_single_cyc(self.x2, self.y2, self.inputadd2, use_2nd=True)            
            
            if self.config == 0:  # photometry-only
                cost_recon = cost_recon1 + cost_recon2
                cost_ycontra = cost_ycontra1 + cost_ycontra2
                        
                lmse_p = self.exp_minus_lmse([latent1, latent2], [latent1_, latent2_])
                lmse_n = self.exp_minus_lmse([latent1], [latent2])
                
            elif self.config == 1:  # image-based
                cost_recon_morphaug1, cost_ycontra_morphaug1, latent_morph1 = self.get_outputs_aug(self.x_morph, self.inputadd, ylogits1, use_2nd=True)
                cost_recon_morphaug2, cost_ycontra_morphaug2, latent_morph2 = self.get_outputs_aug(self.x2_morph, self.inputadd2, ylogits2, use_2nd=True)
                    
                cost_recon = cost_recon1 + cost_recon2 + cost_recon_morphaug1 + cost_recon_morphaug2
                cost_ycontra = cost_ycontra1 + cost_ycontra2 + cost_ycontra_morphaug1 + cost_ycontra_morphaug2
                        
                lmse_p = self.exp_minus_lmse([latent1, latent1, latent2, latent2], [latent1_, latent_morph1, latent2_, latent_morph2])
                lmse_n = self.exp_minus_lmse([latent1], [latent2])

            cost_ce = cost_ce1 + cost_ce2
            cost_lcontra = -1 * tf.log(lmse_p / (lmse_p + lmse_n))
        #    print (cost_lcontra)
                
            p_set = [p11]
            ce_set = [cost_ce_single11]
            latent_set = [latent1[:, :self.dim_latent_main]]
            return cost_ce, cost_recon, cost_lcontra, cost_ycontra, p_set, ce_set, latent_set



