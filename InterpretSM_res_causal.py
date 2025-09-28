import numpy as np
import time
import argparse



def compute_corr(prop1, prop2, weights=0):
    '''
    Compute correlations between two properties/variables. 
    Input shape (of each property/variable): [n_sample, k_nn]
    Output shape: [n_sample]
    '''
    
    filt = (abs(prop1) < 9000) & (abs(prop2) < 9000)  # filter out null values
    if np.sum(weights) == 0:
        mask = np.zeros(prop1.shape)
        mask[filt] = 1.0
    else:
        mask = np.zeros(prop1.shape)
        mask[filt] = weights[filt]
    prop1mean = np.sum(prop1 * mask, 1, keepdims=True) / np.sum(mask, 1, keepdims=True)
    prop2mean = np.sum(prop2 * mask, 1, keepdims=True) / np.sum(mask, 1, keepdims=True)
    dprop1 = prop1 - prop1mean
    dprop2 = prop2 - prop2mean
    sigprop1 = np.sqrt(np.sum(dprop1 * dprop1 * mask, 1))
    sigprop2 = np.sqrt(np.sum(dprop2 * dprop2 * mask, 1))
    return np.sum(dprop1 * dprop2 * mask, 1) / sigprop1 / sigprop2



def get_corr(prop1, prop2, weights=0):
    '''
    Get correlations between two properties/variables with and without random permutations.
    Input shape (of each property/variable): [n_sample, k_nn]
    Output shape: [n_sample]
    '''
    
    prop1_shuffle = np.ones(prop1.shape) * (-9999)
    n_sample = len(prop1)
    
    if np.sum(weights) == 0:
        for i in range(n_sample):
            filt = (abs(prop1[i]) < 9000) & (abs(prop2[i]) < 9000)  # filter out null values
            prop1_shuffle[i][filt] = np.random.permutation(prop1[i][filt])
        prop2_shuffle = prop2
    else:
        prop2_shuffle = np.ones(prop1.shape) * (-9999)
        for i in range(n_sample):
            filt = (abs(prop1[i]) < 9000) & (abs(prop2[i]) < 9000)
            w_i = weights[i][filt] / np.sum(weights[i][filt])
            if np.sum(w_i) != 1: continue
            prop1_shuffle[i][filt] = np.random.choice(prop1[i][filt], len(prop1[i][filt]), p=w_i)
            prop2_shuffle[i][filt] = np.random.choice(prop2[i][filt], len(prop2[i][filt]), p=w_i)
            
    corr_original = compute_corr(prop1, prop2, weights=weights)
    corr_permuted = compute_corr(prop1_shuffle, prop2_shuffle, weights=0)
    return corr_original, corr_permuted



def compute_2ndregress(prop_y, prop_x):
    '''
    Compute the residual of a property/variable (y) after quadratically regressed on another property/variable (x).
    Input shape (of each property/variable): [n_sample, k_nn]
    Output shape: [n_sample, k_nn]
    '''
    
    prop_y_given_x = np.ones(prop_y.shape) * (-9999)
    
    filt = (abs(prop_y) < 9000) & (abs(prop_x) < 9000)  # filter out null values
    mask = np.zeros(prop_y.shape)
    mask[filt] = 1.0
    
    sum_x = np.sum(prop_x * mask, 1)
    sum_x2 = np.sum((prop_x ** 2) * mask, 1)
    sum_x3 = np.sum((prop_x ** 3) * mask, 1)
    sum_x4 = np.sum((prop_x ** 4) * mask, 1)
    
    sum_y = np.sum(prop_y * mask, 1)
    sum_xy = np.sum(prop_x * prop_y * mask, 1)
    sum_x2y = np.sum((prop_x ** 2) * prop_y * mask, 1)
    
    n = np.sum(mask, 1)
    
    deno = n * sum_x2 * sum_x4 + 2 * sum_x * sum_x2 * sum_x3 - sum_x2 ** 3 - n * sum_x3 ** 2 - sum_x ** 2 * sum_x4
    b0 = (sum_x2 * sum_x4 - sum_x3 ** 2) * sum_y + (sum_x2 * sum_x3 - sum_x * sum_x4) * sum_xy + (sum_x * sum_x3 - sum_x2 ** 2) * sum_x2y
    b1 = (sum_x2 * sum_x3 - sum_x * sum_x4) * sum_y + (n * sum_x4 - sum_x2 ** 2) * sum_xy + (sum_x * sum_x2 - n * sum_x3) * sum_x2y
    b2 = (sum_x * sum_x3 - sum_x2 ** 2) * sum_y + (sum_x * sum_x2 - n * sum_x3) * sum_xy + (n * sum_x2 - sum_x ** 2) * sum_x2y

    a0 = np.expand_dims(b0 / deno, 1)
    a1 = np.expand_dims(b1 / deno, 1)
    a2 = np.expand_dims(b2 / deno, 1)
    
    prop_y_given_x[filt] = (prop_y - a0 - a1 * prop_x - a2 * prop_x ** 2)[filt]
    return prop_y_given_x



def get_pred_efficiency(prop_y, prop_x):
    '''
    Get the predictive efficiency of a property/variable (x) in predicting another property/variable (y).
    Input shape (of each property/variable): [n_sample, k_nn]
    Output shape: [n_sample]
    '''
    
    filt = (abs(prop_y) < 9000) & (abs(prop_x) < 9000)  # filter out null values
    mask = np.zeros(prop_y.shape)
    mask[filt] = 1.0
    prop_y_given_x = compute_2ndregress(prop_y, prop_x)
    
    prop1 = prop_y
    prop2 = prop_y_given_x
    prop1mean = np.sum(prop1 * mask, 1, keepdims=True) / np.sum(mask, 1, keepdims=True)
    prop2mean = np.sum(prop2 * mask, 1, keepdims=True) / np.sum(mask, 1, keepdims=True)
    dprop1 = prop1 - prop1mean
    dprop2 = prop2 - prop2mean
    varprop1 = np.sum(dprop1 * dprop1 * mask, 1)
    varprop2 = np.sum(dprop2 * dprop2 * mask, 1)
    return (varprop1 - varprop2) / varprop1





##### Load data #####


directory = './'
parser = argparse.ArgumentParser()
parser.add_argument('--datalabel', help='label that stands for the input data in supervised contrastive learning', type=str, default='Mopt')

args = parser.parse_args()
datalabel = args.datalabel


# IDs of k nearest neighbors (from the training sample) for all data instances in the training, validation and test samples in sequence.
fknn = np.load(directory + 'idKNN_' + datalabel + '_.npz')
idknn = fknn['id_knn']
n_train = fknn['n_train']
n_validation = fknn['n_validation']
n_test = fknn['n_test']
    
# user-defined catalog that contains stellar mass estimates (also used in "InterpretSM_data.py")
catalog = np.load(directory + 'catalog.npz')
prop_y = catalog['lgm_tot_p50']  # target variable, i.e., stellar mass
prop_xq = catalog['expAB_r']  # query variable, e.g., r-band inclination
prop_xc = catalog['z']  # conditional variable, e.g., spectroscopic redshift
variablelabel = '_qInclination_cZspec'

k_nn = 100  # number of nearest neighbors for each data instance
prop_y = prop_y[idknn[:, :k_nn]]
prop_xq = prop_xq[idknn[:, :k_nn]]
prop_xc = prop_xc[idknn[:, :k_nn]]



##### Get (conditional) correlations and predictive efficiency #####

print ('Start computing...')

# unconditional
corr_original_uncond, corr_permuted_uncond = get_corr(prop_y, prop_xq)
pred_eff_uncond = get_pred_efficiency(prop_y, prop_xq)
    
# conditional    
prop_y_cond = compute_2ndregress(prop_y, prop_xc)
prop_xq_cond = compute_2ndregress(prop_xq, prop_xc)

corr_original_cond, corr_permuted_cond = get_corr(prop_y_cond, prop_xq_cond)
pred_eff_cond = get_pred_efficiency(prop_y_cond, prop_xq_cond)


print ('Start saving...')

np.savez(directory + 'results_causal_' + datalabel + variablelabel, n_train=n_train, n_validation=n_validation, n_test=n_test,
         corr_original_uncond=corr_original_uncond, corr_permuted_uncond=corr_permuted_uncond, pred_eff_uncond=pred_eff_uncond,
         corr_original_cond=corr_original_cond, corr_permuted_cond=corr_permuted_cond, pred_eff_cond=pred_eff_cond)
        