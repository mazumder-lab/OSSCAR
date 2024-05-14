# OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization

This is the offical repo of the ICML 2024 paper **OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization**


## Requirements: 
This code has been tested with Python 3.9.6 and the following packages:
+ torch 2.0.0
+ transformers 4.35.2
+ datasets 2.15.0
+ numpy 1.24.3

## Datasets and models

The data files can be found at https://www.dropbox.com/scl/fi/6xg1voa7go9x2uds1y2mq/scd_data.zip?rlkey=8bwzshamiyvcvpd146ymkp0vc&dl=0.

To download the model, set cached to False in get_opt (opt_prune.py) and run the script. After the model is downloaded, upload the model files to the cluster and set cached to True. 


## Running:

python opt_prune.py facebook/opt-125m c4 0.5 --algo OSSCAR_prune --model_path {your_path} --data_path {your_path}

We usually use c4 as the training (calibration) data and report perplexity on wikitext2. Here 0.5 denotes the sparsity.

## Citing OSSCAR:

If you find OSSCAR useful in your research, please consider citing the following paper.

```
@article{meng2024osscar,
  title={OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization},
  author={Meng, Xiang and Ibrahim, Shibal and Behdin, Kayhan and Hazimeh, Hussein and Ponomareva, Natalia and Mazumder, Rahul},
  journal={arXiv preprint arXiv:2403.12983},
  year={2024}
}
```
