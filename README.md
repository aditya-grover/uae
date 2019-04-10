
Uncertainty Autoencoders
============================================

This repository provides a reference implementation for learning uncertainty autoencoders as described in the paper:

> Uncertainty Autoencoders: Learning Compressed Representations via Variational Information Maximization 
Aditya Grover, Stefano Ermon  
International Conference on Artificial Intelligence and Statistics (AISTATS), 2019 
Paper: https://arxiv.org/abs/1807.01442

## Requirements

The codebase is implemented in Python 3.6 and Tensorflow. To install the necessary requirements, run the following commands:

```
pip install -r requirements.txt
```

## Options

Learning and inference of Uncertainty Autoencoders is handled by the `main.py` script which provides the following command line arguments.

```
  --pretrained-model-dir PRETRAINED_MODEL_DIR
                        Directory containing pretrained model
  --dataset DATASET     Dataset to use
  --input-type INPUT_TYPE
                        Where to take input from
  --input-path-pattern INPUT_PATH_PATTERN
                        Pattern to match to get images
  --num-input-images NUM_INPUT_IMAGES
                        number of input images
  --batch-size BATCH_SIZE
                        How many examples are processed together
  --measurement-type MEASUREMENT_TYPE
                        measurement type
  --noise-std NOISE_STD
                        std dev of noise
  --num-measurements NUM_MEASUREMENTS
                        number of gaussian measurements
  --model-types MODEL_TYPES [MODEL_TYPES ...]
                        model(s) used for estimation
  --mloss1_weight MLOSS1_WEIGHT
                        L1 measurement loss weight
  --mloss2_weight MLOSS2_WEIGHT
                        L2 measurement loss weight
  --zprior_weight ZPRIOR_WEIGHT
                        weight on z prior
  --dloss1_weight DLOSS1_WEIGHT
                        -log(D(G(z))
  --dloss2_weight DLOSS2_WEIGHT
                        log(1-D(G(z))
  --sparse_gen_weight SPARSE_GEN_WEIGHT
                        weight for sparse deviations
  --optimizer-type OPTIMIZER_TYPE
                        Optimizer type
  --learning-rate LEARNING_RATE
                        learning rate
  --momentum MOMENTUM   momentum value
  --max-update-iter MAX_UPDATE_ITER
                        maximum updates to z
  --num-random-restarts NUM_RANDOM_RESTARTS
                        number of random restarts
  --decay-lr            whether to decay learning rate
  --lmbd LMBD           lambda : regularization parameter for LASSO
  --lasso-solver LASSO_SOLVER
                        Solver for LASSO
  --const_dummy CONST_DUMMY
                        dummy hack
  --save-images         whether to save estimated images
  --save-stats          whether to save estimated images
  --print-stats         whether to print statistics
  --checkpoint-iter CHECKPOINT_ITER
                        checkpoint every x batches
  --image-matrix IMAGE_MATRIX
                        0 = 00 = no image matrix, 1 = 01 = show image matrix 2
                        = 10 = save image matrix 3 = 11 = save and show image
                        matrix

```

## Examples

 

** NOTE: ** An experimental reimplementation is available in pytorch_src/ folder. Use at own risk.


## Citing

If you find Uncertainty Autoencoders useful in your research, please consider citing the following paper:


>@inproceedings{grover2019uncertainty,  
  title={Uncertainty Autoencoders: Learning Compressed Representations via Variational Information Maximization},  
  author={Grover, Aditya and Ermon, Stefano},  
  booktitle={International Conference on Artificial Intelligence and Statistics},  
  year={2019}}