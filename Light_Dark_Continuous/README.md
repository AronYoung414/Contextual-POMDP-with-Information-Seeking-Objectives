# Contextual-POMDP-with-Information-Seeking-Objectives
This is the project for the paper "C-IDS: Solving Contextual POMDP via Information-Directed Objective"

To reproduce our results, simply run the file **main.py**. 

You may also change the environment setting in the file **light_dark_environment.py**

VPG_EKF.py is the algorithm file for the variational policy gradient method.

policy_continuous.py is the LSTM policy for the parametrization.

policy_eva generates trajectories to evaluate the policy.

Run POMCP_baseline.py for the POMCP baseline comparison.

Run train_pg_solver.py for RDPG-RNN  baseline comparison. 

compare_three_methods.py is for generating policies under the same environment for three methods.

build_pomcp_cached_policy.py is file to provide APIs for the policy comparison.

All the code for plotting pictures are in the folder "plot_files". And you can check all the data we have here for 
different inference mechanism. data_ekf is for our current methods. Other data are from some failed method. For example,
mg means Gaussian mixture.