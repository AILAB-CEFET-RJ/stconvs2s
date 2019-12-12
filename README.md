# STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for weather forecasting

This repository has the open source implementation of a new architecture termed STConvS2S. To sum up, our approach (STConvS2S) uses only convolutional neural network (CNN) to tackle the sequence-to-sequence task using spatiotemporal data. We compare our results with ConvLSTM architecture. Details in the paper: [arXiv:1912.00134](https://arxiv.org/abs/1912.00134).

![STConvS2S architecture](/image/stconvs2s.png)

## Requirements

Mainly, our code uses Python 3.6 and PyTorch 1.0. See [config/environment.yml](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/config/environment.yml) for other requirements.

To install packages with the same version as we executed our experiments, run the code below:

```
cd config
./create-env.sh
```

## Datasets

All datasets are publicly available at http://doi.org/10.5281/zenodo.3558773. Datasets must be placed in the [data](https://github.com/MLRG-CEFET-RJ/stconvs2s/tree/master/data) folder.

Note: In the *15-step ahead* datasets, just for the convenience of having the input and targe tensors in the same file, we define them in the same shape
(# of samples, **15**, # of latitude, # of longitude, 1). However, in the code, we force the correct length of the input sequence in the tensor as shown [here](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/tool/dataset.py#L28).

## Experiments

Jupyter notebooks for the first sequence-to-sequence task (given the previous 5 grids, predict the next 5 grids) can be found in the [experiments](https://github.com/MLRG-CEFET-RJ/stconvs2s/tree/master/experiments) folder (see Table 1 in the paper).


Final experiments (see Table 2 in the paper) compare STConvS2S (our architecture) with ConvLSTM architecture and a baseline (ARIMA models). We evaluate the models in two horizons: 5 and 15-steps ahead. This task is performed using [main.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/main.py) (for deep learning models) and [baseline.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/baseline.py) (for ARIMA models). Results can be found in the [output](https://github.com/MLRG-CEFET-RJ/stconvs2s/tree/master/output) folder.


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
python main.py -i 10 -v 4 --plot > output/full-dataset/results/cfsr-stconvs2s-rmse-v4.out
```

The above command executes STConvS2S in 10 iterations (`-i`), indicating the model version (`-v`), allowing the generation of plots in the training phase (`--plot`).

### ConvLSTM

To run experiments with ConvLSTM architecture, add the `--convlstm` parameter to the above command.


### Baseline

```
python baseline.py > output/baseline/cfsr-arima.out
```

### Additional parameters

* add `--chirps`: change the dataset to rainfall (CHIRPS). Default dataset: temperature (CFSR). 
* add `--mae`: change the metric to MAE. Default metric: RMSE.
* add `-s 15`: change the horizon. Default horizon: 5.
* add `--email`: send email at the end. To use this functionality, set your email in the file [config/mail_config.ini](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/config/mail_config.ini).

## Citation
```
@article{Nascimento2019,
	author    = {Rafaela C. Nascimento, Yania M. Souto, Eduardo Ogasawara, Fabio Porto and Eduardo Bezerra},
	title     = {{STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for weather forecasting}},
	journal   = {arXiv:1912.00134},
	year      = {2019}
}
```

## Contact
To give your opinion about this work, send an email to `rafaela.nascimento@eic.cefet-rj.br`.