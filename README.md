# STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for weather forecasting

This repository has the open source implementation of a new architecture, termed STConvS2S. To sum up, our approach (STConvS2S) use only convolutional neural network (CNN) to tackle spatiotemporal forecasting tasks. We compare our results with ConvLSTM architecture, considered the state-of-art. Details in the paper: [arXiv:0000.0000](arxiv.org)

![STConvS2S architecture](/image/abstraction-seq2seq.png)

## Requirements

Mainly, our code uses Python 3.6 and PyTorch 1.0. See `/config/environment.yml` for other requirements.

To install packages with the same version as we execute our experiments, run the code below:

```
cd config
./create-env.sh
```

## Datasets

All datasets are publicly available for download on Zenodo (open source data repository). Datasets must be placed in the `/data` folder.

## Experiments

Jupyter notebooks for the first sequence-to-sequence task (given the previous 5 grids, predict the next 5 grids) can be found in the `/experiments` folder (see Table 1 in the paper).


Final experiments (see Table 2 in the paper) compare STConvS2S (our architecture) with state-of-art architecture and a baseline (ARIMA models). We evaluate the models in two horizons: 5 and 15-steps ahead. This task is performed using `main.py` (for deep learning models) and `baseline.py` (for ARIMA models). Results can be found in the `/output` folder.


* `/output/full-dataset` (for deep learning models)
	* `/checkpoints`: pre-trained models that allow you to recreate the training configuration (weights, loss, optimizer).
	* `/losses`: training and validation losses. Can be used to recreate the error analysis plot
	* `/plots`:	error analysis plots from training phase
	* `/results`: evaluation phase results with metric value (RMSE or MAE), training time, best epochs and so on.

* `/output/baseline`: results of baseline models
	

## Usage

First load the conda environment with the installed packages.

```
source activate pytorch
```

Below are examples of how to run each model.

### STConvS2S
```
python main.py -i 10 -v 4 --plot --email > output/full-dataset/results/cfsr-stconvs2s-rmse-v4.out
```

The above command executes STConvS2S in 10 iterations (`-i`), indicating the model version (`-v`), allowing the generation of plots in the training phase (`--plot`) and sending email at the end (`--email`).

### ConvLSTM

To run experiments with ConvLSTM architecture, add the `--convlstm` parameter to the above command.


### Baseline

```
python baseline.py > output/baseline/cfsr-arima.out
```

### Adicional parameters

* add `--chirps`: changes the dataset to rainfall (CHIRPS). Default dataset: temperature (CFSR). 
* add `--mae`: changes the metric to MAE. Default metric: RMSE
* add `-s 15`: changes the horizon. Default horizon: 5
