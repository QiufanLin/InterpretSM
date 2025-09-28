import numpy as np
import time
import argparse



##### Load data #####


directory = './'
parser = argparse.ArgumentParser()
parser.add_argument('--datalabel', help='label that stands for the input data in supervised contrastive learning', type=str, default='Mopt')

args = parser.parse_args()
datalabel = args.datalabel


if 'I' in datalabel:  # image-based
    fname = 'output_causalSCL_' + datalabel + '_min6_max12p5_bin520_ADAM_batch64_ite180000_m16e512_coeffRecon100_scratch_ne1_.npz'
else:  # photometry-only   
    fname = 'output_causalSCL_' + datalabel + '_min6_max12p5_bin520_ADAM_batch64_ite180000_m8e8_coeffRecon1_scratch_ne1_.npz'

fscl = np.load(directory + fname)
id_train = fscl['id_train']
n_train = fscl['n_train']
n_validation = fscl['n_validation']
n_test = fscl['n_test']
n_all = n_train + n_validation + n_test 

latent = fscl['latent']
# The latent vectors for the training, validation and test samples are concatenated in sequence.
# Shape: [n_all, dim_latent_main]

latent_train = latent[:n_train]      

print ('Training,Validation,Test:', n_train, n_validation, n_test)
print (latent_train.shape)



##### KNN #####

### Get the IDs of k nearest neighbors (from the training sample) for all data instances in the training, validation and test samples.

print ('Start computing...')

kmax = 2000
id_knn = np.zeros((n_all, kmax))
    
start = time.time()
for i in range(n_all):
    if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
    distsq_i = np.mean((latent[i:i+1] - latent_train) ** 2, 1)       
    arg_knn_i = np.argsort(distsq_i)[:kmax]
    id_knn[i] = id_train[arg_knn_i]    
id_knn = np.cast['int32'](id_knn)


print ('Start saving...')

np.savez('idKNN_' + datalabel + '_', id_knn=id_knn, n_train=n_train, n_validation=n_validation, n_test=n_test)


