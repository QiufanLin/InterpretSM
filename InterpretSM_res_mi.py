import numpy as np
import time
import argparse



def get_mi_components(yt, ce_set):
    ce1, ce2, ce3 = ce_set
    
    '''
    Estimate information components (i.e., redundant, unique, synergistic) given two inputs.
    "yt": target variable, i.e., stellar mass
    "ce1": cross-entropy i.e., minus log-probability given <input 1>
    "ce2": cross-entropy i.e., minus log-probability given <input 2>
    "ce3": cross-entropy i.e., minus log-probability given <input 1 -Union- input 2>
    Shape: [n_sample]
    '''
    
    # Estimate the prior
    y_min = 6.0
    y_max = 12.5
    bins = 520
    ylist = np.arange(bins+1) * (y_max - y_min) / bins + y_min
    
    # Using all data instances from the training, validation and test samples together to estimate the prior distribution, assuming there is no sample mismatch
    yp = np.histogram(yt, bins, (y_min, y_max))[0]
    yp = yp / np.sum(yp)
    
    h_y_indiv = np.zeros(len(yt))
    for i in range(bins):
        filt = (yt >= ylist[i]) & (yt < ylist[i+1])
        h_y_indiv[filt] = -1 * np.log(yp[i])
        
    # Estimate mutual information
    mi1 = h_y_indiv - ce1
    mi2 = h_y_indiv - ce2
    mi3 = h_y_indiv - ce3
    
    redundant = np.min(np.stack([mi1, mi2], 1), 1)
    unique1 = mi1 - redundant
    unique2 = mi2 - redundant
    synergistic = mi3 - (redundant + unique1 + unique2)
    return redundant, synergistic, unique1, unique2





##### Load data #####


directory = './'
parser = argparse.ArgumentParser()
parser.add_argument('--datalabel1', help='label that stands for <input 1> in mutual information estimation', type=str, default='Mopt')
parser.add_argument('--datalabel2', help='label that stands for <input 2> in mutual information estimation', type=str, default='Ioptnorm')
parser.add_argument('--datalabel3', help='label that stands for <input 1 -Union- input 2> in mutual information estimation', type=str, default='MoptUnionIoptnorm')

args = parser.parse_args()
datalabel1 = args.datalabel1
datalabel2 = args.datalabel2
datalabel3 = args.datalabel3


ce_set = []
for datalabel in [datalabel1, datalabel2, datalabel3]:
    if 'I' in datalabel:  # image-based
        fname = 'output_MI_' + datalabel + '_min6_max12p5_bin520_ADAM_batch128_ite180000_m16e0_scratch_ne1_.npz'
    else:  # photometry-only   
        fname = 'output_MI_' + datalabel + '_min6_max12p5_bin520_ADAM_batch128_ite180000_m8e0_scratch_ne1_.npz'

    fmi = np.load(directory + fname)
    ce = fmi['cross_entropy_indiv']  # The cross-entropy estimates for the training, validation and test samples are concatenated in sequence.
    ce_set.append(ce)

id_train = fmi['id_train']
id_validation = fmi['id_validation']
id_test = fmi['id_test']
id_all = np.concatenate([id_train, id_validation, id_test])

n_train = fmi['n_train']
n_validation = fmi['n_validation']
n_test = fmi['n_test']

# user-defined catalog that contains stellar mass estimates (also used in "InterpretSM_data.py")
catalog = np.load(directory + 'catalog.npz')
yt = catalog['lgm_tot_p50'][id_all]  # stellar mass



##### Get information components #####

print ('Start computing...')

redundant, synergistic, unique1, unique2 = get_mi_components(yt, ce_set)


print ('Start saving...')

np.savez(directory + 'results_mi_' + datalabel + '_', n_train=n_train, n_validation=n_validation, n_test=n_test,
         redundant=redundant, synergistic=synergistic, unique1=unique1, unique2=unique2)
        