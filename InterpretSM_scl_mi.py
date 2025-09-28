import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
from InterpretSM_network import *
from InterpretSM_data import *
import threading
import argparse



##### Settings #####


directory = './'

parser = argparse.ArgumentParser()
parser.add_argument('--ne', help='# experiment', type=int, default=1)
parser.add_argument('--phase', help='0: conducting training; 1: conducting inference', type=int)
parser.add_argument('--texp', help='select an experiment type (0: mutual information estimation; 1: supervised contrastive learning for causal analysis)', type=int)
parser.add_argument('--id_dataset', help='select an input data set', type=int)
parser.add_argument('--bins', help='number of bins for expressing stellar mass probability densities', type=int, default=520)

args = parser.parse_args()
ne = args.ne
phase = args.phase
texp = args.texp
id_dataset = args.id_dataset
bins = args.bins


learning_rate_ini = 0.0001
learning_rate_step = 60000  # reduce the learning rate at # iterations during training
lr_reduce_factor = 5.0
ite_val_print = 10000  # print validation results per # iterations during training    

iterations = 180000  # total number of iterations for training
ite_point_save = [i for i in range(120000, 180000+10000, 10000)]  # save model at # iterations during training


y_min = 6.0  # lower bound of stellar mass
y_max = 12.5  # upper bound of stellar mass
wbin = (y_max - y_min) / bins  # bin width == 0.0125 (bins=520)


if texp == 0:  # mutual information estimation
    batch_train = 128  # size of a mini-batch for training

elif texp == 1:  # supervised contrastive learning for causal analysis
    batch_train = 64  # size of a mini-batch for training
    
    
repetition_per_ite = 1
use_cpu = False
num_threads = 4 + 1



###### Define input data #####


optical_bands = ['u', 'g', 'r', 'i', 'z']
infrared_bands = ['w1', 'w2', 'w3']
all_bands = optical_bands + infrared_bands
img_norm = False  # normalize optical images with Petrosian fluxes in each band

# "input_keys": main input variables from a user-defined catalog or multi-band cutout images (specified in "InterpretSM_data.py"), e.g., Petrosian magnitudes in the five optical bands
# "inputadd_keys": additional input variables from the user-defined catalog (specified in "InterpretSM_data.py"), e.g., galactic reddening E(B-V) and Petrosian magnitude errors in the five optical bands

if texp == 1:  # supervised contrastive learning for causal analysis
    # id_dataset == 1, 2, 3, 4 or 5, corresponding to five imodels
    
    if id_dataset == 1:  # optical photometry
        Dataset = 'Mopt_'
        input_keys = ['petroMag_' + b for b in optical_bands]
        inputadd_keys = ['EBV'] + ['petroMagErr_' + b for b in optical_bands]

    elif id_dataset == 2:  # optical & infrared photometry
        Dataset = 'MoptMir_'
        input_keys = ['petroMag_' + b for b in optical_bands] + [b + 'mpro' for b in infrared_bands]
        inputadd_keys = ['EBV'] + ['petroMagErr_' + b for b in optical_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors

    elif id_dataset == 3:  # optical images
        Dataset = 'Iopt_'
        input_keys = ['img_' + b for b in optical_bands]
        inputadd_keys = ['EBV']

    elif id_dataset == 4:  # optical images & infrared photometry
        Dataset = 'IoptMir_'
        input_keys = ['img_' + b for b in optical_bands]
        inputadd_keys = ['EBV'] + [b + 'mpro' for b in infrared_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors
        
    elif id_dataset == 5:  # optical images, infrared photometry & spectroscopic redshift
        Dataset = 'IoptMirZspec_'
        input_keys = ['img_' + b for b in optical_bands]
        inputadd_keys = ['EBV', 'z'] + [b + 'mpro' for b in infrared_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors

    else:
        raise ValueError('Invalid <id_dataset>')
        
        
elif texp == 0:  # mutual information estimation
    # id_dataset == an integer in the form of "abc"
    d_c = id_dataset % 10
    d_b = int((id_dataset % 100 - d_c) / 10)
    d_a = int((id_dataset - d_c - 10 * d_b) / 100)
    
    if d_a not in [1, 2, 3, 4, 5]:
        raise ValueError('Invalid <id_dataset>')
        
    if d_a == 1:  # different data modalities
        # d_b == 1, 2, 3, 4 or 5, corresponding to five input data modalities
        # d_c == 1, 2 or 3, corresponding to "input 1", "input 2", or "input 1 -Union- input 2", respectively
        
        if d_b not in [1, 2, 3, 4, 5] or d_c not in [1, 2, 3]:
            raise ValueError('Invalid <id_dataset>')
            
        if d_b == 1:  # <optical photometry, normalized optical images>
            Dataset1 = 'Mopt'
            Dataset2 = 'Ioptnorm'
            input_keys1 = ['petroMag_' + b for b in optical_bands] + ['petroMagErr_' + b for b in optical_bands]
            input_keys2 = ['img_' + b for b in optical_bands]
            inputadd_keys1 = inputadd_keys2 = ['EBV']
            img_norm = True
            
        elif d_b == 2:  # <optical photometry, optical morphological parameters>
            Dataset1 = 'Mopt'
            Dataset2 = 'Morph'
            input_keys1 = ['petroMag_' + b for b in optical_bands] + ['petroMagErr_' + b for b in optical_bands]
            input_keys2 = ['Sersic_n_' + b for b in optical_bands] + ['expAB_' + b for b in optical_bands] + ['petroR50_' + b for b in optical_bands] + ['petroR90_' + b for b in optical_bands]
            inputadd_keys1 = inputadd_keys2 = ['EBV']
            
        elif d_b == 3:  # <optical photometry, infrared photometry>
            Dataset1 = 'Mopt'
            Dataset2 = 'Mir'
            input_keys1 = ['petroMag_' + b for b in optical_bands] + ['petroMagErr_' + b for b in optical_bands]
            input_keys2 = [b + 'mpro' for b in infrared_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors
            inputadd_keys1 = inputadd_keys2 = ['EBV']
            
        elif d_b == 4:  # <optical images, infrared photometry>
            Dataset1 = 'Iopt'
            Dataset2 = 'Mir'
            input_keys1 = ['img_' + b for b in optical_bands]
            input_keys2 = [b + 'mpro' for b in infrared_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors
            inputadd_keys1 = inputadd_keys2 = ['EBV']
            
        elif d_b == 5:  # <optical images -Union- infrared photometry, spectroscopic redshift>
            Dataset1 = 'IoptMir'
            Dataset2 = 'Zspec'
            input_keys1 = ['img_' + b for b in optical_bands]
            input_keys2 = ['z']
            inputadd_keys1 = ['EBV'] + [b + 'mpro' for b in infrared_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors
            inputadd_keys2 = ['EBV']
        
        if d_c == 1:  # input 1
            Dataset = Dataset1 + '_'
            input_keys = input_keys1
            inputadd_keys = inputadd_keys1
            
        elif d_c == 2:  # input 2
            Dataset = Dataset2 + '_'
            input_keys = input_keys2
            inputadd_keys = inputadd_keys2
            
        elif d_c == 3:  # input 1 -Union- input 2
            Dataset = Dataset1 + 'Union' + Dataset2 + '_'
            if d_b == 1:
                input_keys = input_keys2
                inputadd_keys = input_keys1 + inputadd_keys1
            else:
                input_keys = input_keys1
                inputadd_keys = input_keys2 + inputadd_keys1
            
           
    elif d_a == 2:  # separating out images in one band
        # d_b == 1, 2, 3, 4 or 5, corresponding to u, g, r, i or z band, respectively
        # d_c == 1, 2 or 3, corresponding to "input 1", "input 2", or "input 1 -Union- input 2", respectively
        
        if d_b not in [1, 2, 3, 4, 5] or d_c not in [1, 2, 3]:
            raise ValueError('Invalid <id_dataset>')
            
        input_keys = ['img_' + b for b in optical_bands]
        inputadd_keys = ['EBV']
        
        if d_c == 1:  # using the single band that is separated out
            Dataset = 'I' + optical_bands[d_b-1] + '_'
            input_keys = input_keys[d_b-1:d_b]
        
        elif d_c == 2:  # using the remaining four bands
            bands_remain = optical_bands[:d_b-1] + optical_bands[d_b:]
            Dataset = 'I'
            for b in bands_remain:
                Dataset = Dataset + b
            Dataset = Dataset + '_'
            input_keys = input_keys[:d_b-1] + input_keys[d_b:]
            
        elif d_c == 3:  # using all the five optical bands
            Dataset = 'Iopt_'
            
            
    elif d_a == 3:  # separating out images in two adjacent bands
        # d_b == 1, 2, 3 or 4, corresponding to ug, gr, ri or iz bands, respectively
        # d_c == 1, 2 or 3, corresponding to "input 1", "input 2", or "input 1 -Union- input 2", respectively
        
        if d_b not in [1, 2, 3, 4] or d_c not in [1, 2, 3]:
            raise ValueError('Invalid <id_dataset>')
            
        input_keys = ['img_' + b for b in optical_bands]
        inputadd_keys = ['EBV'] 
        
        if d_c == 1:  # using the two bands that are separated out
            Dataset = 'I' + optical_bands[d_b-1] + optical_bands[d_b] + '_'
            input_keys = input_keys[d_b-1:d_b+1]
        
        elif d_c == 2:  # using the remaining three bands
            bands_remain = optical_bands[:d_b-1] + optical_bands[d_b+1:]
            Dataset = 'I'
            for b in bands_remain:
                Dataset = Dataset + b
            Dataset = Dataset + '_'
            input_keys = input_keys[:d_b-1] + input_keys[d_b+1:]
            
        elif d_c == 3:  # using all the five optical bands
            Dataset = 'Iopt_'
            
            
    elif d_a == 4:  # separating out photometry in one band
        # d_b == 1, 2, 3, 4, 5, 6, 7 or 8, corresponding to u, g, r, i, z, W1, W2 or W3 band, respectively
        # d_c == 1, 2 or 3, corresponding to "input 1", "input 2", or "input 1 -Union- input 2", respectively

        if d_b not in [1, 2, 3, 4, 5, 6, 7, 8] or d_c not in [1, 2, 3]:
            raise ValueError('Invalid <id_dataset>')
            
        input_keys = ['petroMag_' + b for b in optical_bands] + [b + 'mpro' for b in infrared_bands]
        inputadd_keys = ['petroMagErr_' + b for b in optical_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors

        if d_c == 1:  # using the single band that is separated out
            Dataset = 'M' + all_bands[d_b-1] + '_'
            input_keys = input_keys[d_b-1:d_b]
            inputadd_keys = ['EBV'] + inputadd_keys[d_b-1:d_b]
            
        elif d_c == 2:  # using the remaining seven bands
            bands_remain = all_bands[:d_b-1] + all_bands[d_b:]
            Dataset = 'M'
            for b in bands_remain:
                Dataset = Dataset + b
            Dataset = Dataset + '_'
            input_keys = input_keys[:d_b-1] + input_keys[d_b:]
            inputadd_keys = ['EBV'] + inputadd_keys[:d_b-1] + inputadd_keys[d_b:]
            
        elif d_c == 3:  # using all the eight bands
            Dataset = 'MoptMir_'            
            inputadd_keys = ['EBV'] + inputadd_keys
            
            
    elif d_a == 5:  # separating out photometry in two adjacent bands
        # d_b == 1, 2, 3, 4, 5, 6 or 7, corresponding to ug, gr, ri, iz, zW1, W1W2, W2W3 bands, respectively
        # d_c == 1, 2 or 3, corresponding to "input 1", "input 2", or "input 1 -Union- input 2", respectively
        
        if d_b not in [1, 2, 3, 4, 5, 6, 7] or d_c not in [1, 2, 3]:
            raise ValueError('Invalid <id_dataset>')
            
        input_keys = ['petroMag_' + b for b in optical_bands] + [b + 'mpro' for b in infrared_bands]
        inputadd_keys = ['petroMagErr_' + b for b in optical_bands] + [b + 'sigmpro' for b in infrared_bands[:2]]  # excluding W3-band magnitude errors
        
        if d_c == 1:  # using the two bands that are separated out
            Dataset = 'M' + all_bands[d_b-1] + all_bands[d_b] + '_'
            input_keys = input_keys[d_b-1:d_b+1]
            inputadd_keys = ['EBV'] + inputadd_keys[d_b-1:d_b+1]
        
        elif d_c == 2:  # using the remaining six bands
            bands_remain = all_bands[:d_b-1] + all_bands[d_b+1:]
            Dataset = 'M'
            for b in bands_remain:
                Dataset = Dataset + b
            Dataset = Dataset + '_'
            input_keys = input_keys[:d_b-1] + input_keys[d_b+1:]
            inputadd_keys = ['EBV'] + inputadd_keys[:d_b-1] + inputadd_keys[d_b+1:]
            
        elif d_c == 3:  # using all the eight bands
            Dataset = 'MoptMir_'            
            inputadd_keys = ['EBV'] + inputadd_keys
    

c_input = len(input_keys)  # number of channels of the main input
c_inputadd = len(inputadd_keys)  # number of channels of the additional input    


if img_norm:  # normalize optical images with Petrosian fluxes in each band
    norm_keys = ['petroFlux_' + b for b in optical_bands]
else:
    norm_keys = None
    

if 'I' in Dataset:
    config = 1  # image-based
    img_size = 64  # spatial size of input images
    dim_latent_main = 16  # dimension of the latent vector that encodes stellar mass information (i.e., input of the estimator)
    dim_latent_ext = 512  # dimension of the other latent vector that encodes other information
    coeff_recon = 100  # coefficient of the reconstruction loss term
else:
    config = 0  # photometry-only   
    img_size = None
    dim_latent_main = 8
    dim_latent_ext = 8
    coeff_recon = 1
    
    
if texp == 0:
    dim_latent_ext = 0  # no need for the other latent vector for mutual information estimation


print ('Input:', input_keys)
print ('InputAdd:', inputadd_keys)
#print ('config:', config)

    

###### Model load/save paths #####

    
if texp == 0:  # mutual information estimation             
    ExType = 'MI_'
elif texp == 1:  # supervised contrastive learning for causal analysis                
    ExType = 'causalSCL_'

Dataset = Dataset + 'min6_max12p5_bin' + str(bins) + '_'

Algorithm = 'ADAM_'

if repetition_per_ite != 1: Algorithm = Algorithm + 'rep' + str(repetition_per_ite) + '_'
Algorithm = Algorithm + 'batch' + str(batch_train) + '_'
Algorithm = Algorithm + 'ite' + str(iterations) + '_'
Algorithm = Algorithm + 'm' + str(dim_latent_main) + 'e' + str(dim_latent_ext) + '_'

if texp == 1:  # supervised contrastive learning for causal analysis                 
    Algorithm = Algorithm + 'coeffRecon' + str(coeff_recon) + '_'

Algorithm = Algorithm + 'scratch_'  
    
InfoAdd = 'ne' + str(ne) 

fx = ExType + Dataset + Algorithm + InfoAdd + '_'

model_savepath = directory + 'model_' + fx + '/'
output_savepath = directory + 'output_' + fx

if phase == 0:  # training
    fi = open(directory + 'f_' + fx + '.txt', 'a')
    fi.write(fx + '\n\n')
    fi.close()

elif phase == 1:  # inference
    model_loadpath = directory + 'model_' + fx + '/'
    iterations = 0

print ('#####') 
print (fx)
print ('#####')
    
    

##### Load data #####


getdata = GetData(directory=directory, texp=texp, config=config, dim_latent_main=dim_latent_main, img_size=img_size, norm_keys=norm_keys,
                  input_keys=input_keys, inputadd_keys=inputadd_keys, c_input=c_input, bins=bins, y_min=y_min, wbin=wbin, phase=phase)

n_train = getdata.n_train
n_validation = getdata.n_validation
n_test = getdata.n_test
id_train = getdata.id_train
id_validation = getdata.id_validation
id_test = getdata.id_test

if phase == 0:  # load data for printing validation results during training
    x_validation, yt_validation, y_validation, inputadd_validation = getdata.get_batch_data(id_=[])



##### Network & Cost Function #####

   
model = Model(texp=texp, config=config, dim_latent_main=dim_latent_main, dim_latent_ext=dim_latent_ext, img_size=img_size, 
              c_input=c_input, c_inputadd=c_inputadd, bins=bins, name='model')

if texp == 0:  # mutual information estimation
    p_set, ce_set, latent_set = model.get_outputs()
    cost = ce_set[0]
    
elif texp == 1:  # supervised contrastive learning for causal analysis
    cost_ce, cost_recon, cost_lcontra, cost_ycontra, p_set, ce_set, latent_set = model.get_outputs()
    cost_recon = coeff_recon * cost_recon
    cost = cost_ce + cost_recon + cost_lcontra + cost_ycontra
        
lr = model.lr

x = model.x
inputadd = model.inputadd
y = model.y
    
x2 = model.x2
inputadd2 = model.inputadd2
y2 = model.y2

if config == 1:  # image-based
    x_morph = model.x_morph
    x2_morph = model.x2_morph



##### Session, saver / optimizer #####


if use_cpu: session_conf = tf.ConfigProto(device_count={'GPU':0})#log_device_placement=True
else:
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)

if phase == 0:  # training
    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)         
    optimizer = optimizer.minimize(cost, var_list=tvars)
    session.run(tf.global_variables_initializer())
    
elif phase == 1:  # inference
    tvars = tf.trainable_variables()
    saver = tf.train.Saver(var_list=tvars)
    saver.restore(session, tf.train.latest_checkpoint(model_loadpath))



##### Training #####


batch_train_ini = batch_train
if texp == 1:
    batch_train = int(batch_train/2)
# For supervised contrastive learning, two different sets of data instances will be loaded as contrastive pairs at a time in a mini-batch, thus the "batch_train" halves.

    
def Train(i, th):
    global x_, y_, inputadd_, x2_, y2_, inputadd2_, x_morph_, x2_morph_
    global running
    
    if th == 0:
        feed_dict = {x:x_, y:y_, inputadd:inputadd_, lr:learning_rate}
        if texp == 1:  # supervised contrastive learning for causal analysis
            feed_dict.update({x2:x2_, y2:y2_, inputadd2:inputadd2_})
            if config == 1:  # image-based
                feed_dict.update({x_morph:x_morph_, x2_morph:x2_morph_})
                
        if i == 0 or (i + 1) % ite_val_print == 0:
            ss = '\n' + 'iteration:' + str(i+1) + ' lr:' + str(learning_rate) + ' mini-batch:' + str(batch_train_ini) + ' time:' + str((time.time() - start) / 60) + ' minutes'
            print (ss)
            fi = open(directory + 'f_' + fx + '.txt', 'a')
            fi.write(ss + '\n')

            if texp == 0:  # mutual information estimation
                cost_train = session.run(cost, feed_dict = feed_dict)
                print ('cost_training (single ce):', cost_train)
                fi.write('cost_training (single ce):' + str(cost_train) + '\n')
            elif texp == 1:  # supervised contrastive learning for causal analysis
                cost_train1 = session.run(ce_set, feed_dict = feed_dict)
                cost_train2 = session.run([cost_ce, cost_recon, cost_lcontra, cost_ycontra], feed_dict = feed_dict)
                print ('cost_training (single ce):', cost_train1)
                print ('cost_training (ce, recon, contra, zcontra):', cost_train2)
                fi.write('cost_training (single ce):' + str(cost_train1) + '\n')
                fi.write('cost_training (ce, recon, contra, zcontra):' + str(cost_train2) + '\n')

            outputs_validation = getdata.get_cost_y_stats([x_validation, yt_validation, y_validation, inputadd_validation], session, x, y, inputadd, x2, y2, inputadd2, p_set, ce_set, latent_set)
            print ('outputs_validation:', outputs_validation)
            fi.write('outputs_validation:' + str(outputs_validation) + '\n')
            fi.close()
            
        for t in range(repetition_per_ite):
            session.run(optimizer, feed_dict = feed_dict)
        running = 0
        
    else:
        def read_data(j):
            index_j = np.arange((j-1)*int(batch_train/(num_threads-1)), j*int(batch_train/(num_threads-1)))
            subbatch = len(index_j)
            x_list, y_list, inputadd_list = getdata.get_next_subbatch(subbatch)

            while True:
                if running == 0: break
            
            x_[index_j] = x_list[0][0]
            y_[index_j] = y_list[0]
            inputadd_[index_j] = inputadd_list[0]
            if texp == 1:  # supervised contrastive learning for causal analysis
                x2_[index_j] = x_list[1][0]
                y2_[index_j] = y_list[1]
                inputadd2_[index_j] = inputadd_list[1]
                if config == 1:  # image-based
                    x_morph_[index_j] = x_list[0][1]
                    x2_morph_[index_j] = x_list[1][1]
                              
        for j in range(1, num_threads):
            if th == j:                
                read_data(j)
                   
                
if phase == 0:
    x_list, y_list, inputadd_list = getdata.get_next_subbatch(batch_train)
    
    x_ = x_list[0][0]
    y_ = y_list[0]
    inputadd_ = inputadd_list[0]
    if texp == 1:  # supervised contrastive learning for causal analysis
        x2_ = x_list[1][0]
        y2_ = y_list[1]
        inputadd2_ = inputadd_list[1]
        if config == 1:  # image-based
            x_morph_ = x_list[0][1]
            x2_morph_ = x_list[1][1]
        
    start = time.time()
    print ('Start training...')
       
    for i in range(iterations):
        if i == 0: 
            learning_rate = learning_rate_ini
        if i == learning_rate_step - 1: 
            learning_rate = learning_rate / lr_reduce_factor
        running = 1
        
        threads = []
        for th in range(num_threads):
            t = threading.Thread(target = Train, args = (i, th))
            threads.append(t)
        for th in range(num_threads):
            threads[th].start()
        for th in range(num_threads):
            threads[th].join()
        
        if (i + 1) in ite_point_save:
            saver = tf.train.Saver()
            saver.save(session, model_savepath, i)
            
            

##### Inference #####


if phase == 1:
    cross_entropy_avg, cross_entropy_indiv, latent, yest_mean, yest_mode, yest_median = getdata.get_cost_y_stats([], session, x, y, inputadd, x2, y2, inputadd2, p_set, ce_set, latent_set)
   
    np.savez(output_savepath, n_train=n_train, n_validation=n_validation, n_test=n_test, id_train=id_train, id_validation=id_validation, id_test=id_test,
             yest_mean=yest_mean, yest_mode=yest_mode, yest_median=yest_median, cross_entropy_avg=cross_entropy_avg, cross_entropy_indiv=cross_entropy_indiv, latent=latent)
print ('\n' + fx)
