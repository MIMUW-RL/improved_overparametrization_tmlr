### Code for reproducing experiments reported in the TMLR paper _Improved Overparametrization Bounds for Global Convergence of Stochastic Gradient Descent for Shallow Neural Networks_ by B. Polaczyk and J. Cyranka [TMLR link](https://openreview.net/forum?id=RjZq6W6FoE)

There are two scripts. The first one `grid_search.py` is producing the output data for the grid plot like in Fig.2 in the paper.

It has several command line parameters and can be executed using for example
```
python3 grid_search.py 10 0 BasicNet1L
```
where the first parameter is the input dimension `d0` splits (for parallel computation), the second parameter is the input dimension split to be used (ranging from `0` to the first parameter value). The third parameter (either `BasicNet1L` or `BasicNet2L` defined in `architectures.py`) is the network architecture to be considered (having either single trainable layer, or two trainable layers). Refer to the paper for the details.

Several `csv` output files will be generated including 
* `{modeln}_results_thr_2.5e-03_set{set}.csv` recording the flag of the loss value below `2.5e-03` after `50k` epochs for `10` independent random initializations;
* `{modeln}_avg_final_loss_set{set}.csv` recording the final loss values achieved after `50k` epochs for `10` independent random initializations;
* `{modeln}_avg_corners_set{set}.csv` recording the ratio of parameter values that are found close to the 'corners' of relu regions after the training completion;

The second script `relu_region.py` is producing the output data for the various metrics recorded along the training episodes as shown in Fig. 3 in the paper.

It is executed 
```
python3 relu_region.py BasicNet1L
```
where the parameter (either `BasicNet1L` or `BasicNet2L` defined in `architectures.py`) is the network architecture to be considered (having either single trainable layer, or two trainable layers). The network architecture `d0, d1 & d2` is defined within the script using lists of considered values.

Several `csv` output files will be generated having names `{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_*.csv`, where `modeln` is the considered network, `d0, d1 & d2` are the network parameters, and `*` is either `losses, Hdists, small_preact_cnts, Diff_norms` for each of the metric considered in the paper (reported in Fig. 3).

In case of further questions please reach us through e-mail.
