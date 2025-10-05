# Interpreting deep learning-based stellar mass estimation via causal analysis and mutual information decomposition

We present the code used in our work "Interpreting deep learning-based stellar mass estimation via causal analysis and mutual information decomposition" published at Astronomy & Astrophysics. As the name suggests, we apply two interpretability techniques, i.e., causal analysis and mutual information decomposition, to understand the mechanisms behind deep learning models that are used for the galaxy stellar mass estimation. The causal analysis involves supervised contrastive learning (SCL) and k-nearest neighbor (KNN) procedures. After these procedures, local independence tests are performed to analyze the causal structures between the target variable (i.e., stellar mass) and other variables. The mutual information decomposition is performed to quantify the redundant, unique and synergistic information components between stellar mass and any two given input datasets.

The paper is available at https://arxiv.org/abs/2509.23901.

The code is tested using:
* CPU: Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz
* GPU: Tesla V100 / T4
* Python 3.7.11
* TensorFlow 2.2.0 (developed with TensorFlow 1 but run with TensorFlow 2 by "tf.disable_v2_behavior()"; future versions will be migrated to TensorFlow 2.)


## Causal analysis

1. Supervised contrastive learning

* Apply supervised contrastive learning by running "InterpretSM_scl_mi.py", which requires "InterpretSM_data.py" and "InterpretSM_network.py". The target variable is stellar mass by default. The argument "id_dataset" specifies an input dataset (e.g., "--id_dataset=1" for using optical photometry and galactic reddening as input). Set "--texp=1" for supervised contrastive learning; "--phase=0" for training a model from scratch. The hyperparameters and implementation details (e.g., the number of training iterations, the mini-batch size, and the network architectures, etc.) are all specified in the code.
* Before applying supervised contrastive learning, users should prepare a catalog file (i.e., "catalog.npz") that contains the catalog data required, such as stellar mass, optical and infrared photometry, galactic reddening, and digit IDs for the training, test and validation samples. A separate file that contains multi-band cutout images is also required (i.e., "images.npz"). See "InterpretSM_scl_mi.py" and "InterpretSM_data.py" for more details.

** Example:
> python InterpretSM_scl_mi.py --texp=1 --id_dataset=1 --phase=0

* After training, rerun "InterpretSM_scl_mi.py" by setting "--phase=1" for restoring the trained SCL model and producing inference results for the training, validation and test samples (saved in an output file named "output….npz").

** Example:
> python InterpretSM_scl_mi.py --texp=1 --id_dataset=1 --phase=1

* The default number of bins for expressing stellar mass probability densities is 520. If needed, use the argument "bins" to assign a different value.

** Example:
> python InterpretSM_scl_mi.py --texp=1 --id_dataset=1 --bins=1040 --phase=0

* To run multiple realizations for cross-checks, set "--ne=2", "--ne=3", etc., though in principle only one realization (i.e., "--ne=1") is required for the following procedures.

2. KNN

* Run "InterpretSM_knn.py" to apply KNN for obtaining the IDs of k nearest neighbors (from the training sample) for all data instances in the training, validation and test samples (saved in an output file named "idKNN....npz"). The KNN procedure is applied to the latent vectors produced in supervised contrastive learning for a specified input dataset. The argument "datalabel" denotes the specified input dataset (identical to "Dataset" defined in "InterpretSM_scl_mi.py"). The user-defined catalog and the output .npz file from supervised contrastive learning (for the specified input dataset) are required. The default SCL realization is used (i.e., with "--ne=1" and "--bins=520").

** Example:
> python InterpretSM_knn.py --datalabel='Mopt'

3. Local independence tests

* Run "InterpretSM_res_causal.py" to conduct local independence tests using the KNN result corresponding to a specified input dataset, producing estimates of local correlations and predictive efficiency (both conditional and unconditional) for each data instance in the training, validation and test samples. The results are saved in an output file named "results_causal….npz". The argument "datalabel", same as in the KNN procedure, denotes the input dataset specified in supervised contrastive learning. The user-defined catalog and the output .npz file from KNN are required. The target property/variable is stellar mass by default. Users may use any parameters as a query variable/property ("prop_xq") and a conditional variable/property ("prop_xc"), and rename the "variablelabel" in the code accordingly.

** Example:
> python InterpretSM_res_causal.py --datalabel='Mopt'


## Mutual information decomposition

1. Cross-entropy/mutual information estimation

* Run "InterpretSM_scl_mi.py" to obtain cross-entropy (i.e., minus log-probability) estimates with stellar mass as the target variable and the input data specified by the argument "id_dataset". See the details of "id_dataset" elaborated in the code. Both "InterpretSM_data.py" and "InterpretSM_network.py" are required. Set "--texp=0" for cross-entropy/mutual information estimation; "--phase=0" for training a model from scratch. Same as in supervised contrastive learning, the user-defined catalog and the multi-band cutout images are required. For any tuple of two input datasets, the models with "input 1", "input 2" and "input 1 -Union- input 2" must be trained individually before decomposing mutual information in the next step.

** Example:
> python InterpretSM_scl_mi.py --texp=0 --id_dataset=111 --phase=0

* After training, rerun "InterpretSM_scl_mi.py" by setting "--phase=1" to produce inference results for the training, validation and test samples (saved in an output file named "output….npz").

** Example:
> python InterpretSM_scl_mi.py --texp=0 --id_dataset=111 --phase=1

2. Decomposition

* Run "InterpretSM_res_mi.py" to produce the information components (i.e., redundant, unique and synergistic) for each data instance in the training, validation and test samples, using the cross-entropy estimates corresponding to a specified tuple of two input datasets in the cross-entropy/mutual information estimation. The results are saved in an output file named "results_mi….npz". The arguments "datalabel1", "datalabel2" and "datalabel3" denote the specified input data (corresponding to "input 1", "input 2" and "input 1 -Union- input 2”, respectively, identical to "Dataset" defined in "InterpretSM_scl_mi.py"). The default realization is used (i.e., with "--ne=1" and "--bins=520"). The user-defined catalog is required to retrieve stellar mass to estimate the prior log-probability (or entropy).

** Example:
> python InterpretSM_res_mi.py --datalabel1='Mopt' --datalabel2='Ioptnorm' --datalabel3='MoptUnionIoptnorm'
