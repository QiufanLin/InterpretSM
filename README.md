# Interpreting deep learning-based stellar mass estimation via causal analysis and mutual information decomposition

(To be completed)

We present the code used in our work "Interpreting deep learning-based stellar mass estimation via causal analysis and mutual information decomposition" accepted for publication at Astronomy & Astrophysics. As the name suggests, we apply two interpretability techniques, i.e., **causal analysis** and **mutual information decomposition**, to understand the mechanisms behind deep learning models that are used for galaxy stellar mass estimation. The causal analysis consists of supervised contrastive learning and k-nearest neighbor (KNN) procedures, upon which local independence tests are performed to analyze the causal structures between the target variable (i.e., stellar mass) and other variables. The mutual information decomposition is performed to quantify the redundant, unique and synergistic information components between stellar mass and any two given input data sets.

The code is tested using:
* CPU: Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz
* GPU: Tesla V100 / T4
* Python 3.7.11
* TensorFlow 2.2.0 (developed with TensorFlow 1 but run with TensorFlow 2 by "tf.disable_v2_behavior()"; future versions will be migrated to TensorFlow 2)
