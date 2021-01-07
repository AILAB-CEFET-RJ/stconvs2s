# STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for weather forecasting

> *UPDATE: new code released with changes in our architecture. Please see the [release notes](https://github.com/MLRG-CEFET-RJ/stconvs2s/releases) for details (Nov/2020)*

This repository has the open source implementation of a new architecture termed STConvS2S. To sum up, our approach (STConvS2S) uses only 3D convolutional neural network (CNN) to tackle the sequence-to-sequence task using spatiotemporal data. We compare our results with state-of-the-art architectures. Details in the [article published in Neurocomputing, Elsevier](https://doi.org/10.1016/j.neucom.2020.09.060) ([arXiv](https://arxiv.org/abs/1912.00134) versions).

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
(# of samples, **15**, # of latitude, # of longitude, 1). However, in the code, we force the correct length of the input sequence in the tensor as shown [here](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/tool/dataset.py#L24).

## Experiments

Jupyter notebooks for the first sequence-to-sequence task (given the previous 5 grids, predict the next 5 grids) can be found in the [notebooks](https://github.com/MLRG-CEFET-RJ/stconvs2s/tree/master/notebooks) folder (see Table 1 and 3 in the paper).


Final experiments (see Table 2 in the paper) compare STConvS2S (our architecture) with deep learning state-of-the-art architectures and ARIMA models. We evaluate the models in two horizons: 5 and 15-steps ahead. This task is performed using [main.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/main.py) (for deep learning models) and [arima.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/arima.py) (for ARIMA models). Results can be found in the [output](https://github.com/MLRG-CEFET-RJ/stconvs2s/tree/master/output) folder.


* `/output/full-dataset` (for deep learning models)
	* `/checkpoints`: pre-trained models that allow you to recreate the training configuration (weights, loss, optimizer).
	* `/losses`: training and validation losses. Can be used to recreate the error analysis plot
	* `/plots`:	error analysis plots from training phase
	* `/results`: evaluation phase results with metric value (RMSE and MAE), training time, best epochs and so on.

* `/output/arima` (for ARIMA models)
	

## Usage

First load the conda environment with the installed packages.

```
source activate pytorch
```

Below are examples of how to run each model.

### STConvS2S

We provide two variants of the STConvS2S architecture that satisfy the causal constraint. Each can be executed as follows (complete example of how we performed the exepriments, [here](examples.md))

#### STConvS2S-R (*applies a reverse function in the sequence*)

```
python main.py -i 3 -v 4 -m stconvs2s-r --plot > output/full-dataset/results/cfsr-stconvs2s-rmse-v4.out
```

The above command executes STConvS2S-R (`-m stconvs2s-r`) in 3 iterations (`-i`), indicating the model version (`-v`), allowing the generation of plots in the training phase (`--plot`).

#### STConvS2S-C (*applies causal convolution*)

To run experiments with this architecture, switch to the `-m stconvs2s-c` parameter.


### ARIMA

```
python arima.py > output/arima/cfsr-arima.out
```

### Additional parameters

* add `--chirps`: change the dataset to rainfall (CHIRPS). Default dataset: temperature (CFSR). 
* add `-s 15`: change the horizon. Default horizon: 5.
* add `-l 3 -d 32 -k 5`: define the number of layers (`l`), filters (`d`) and the kernel size (`-k`).
* add `--email`: send email at the end. To use this functionality, set your email in the file [config/mail_config.ini](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/config/mail_config.ini).
* add `--small-dataset`: have a quick training using a few sample dataset.

Check out the other possible parameters [here](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/main.py#L15-L34).

## Citation
```
@article{Castro2021,
	author 	= "Rafaela Castro and Yania M. Souto and Eduardo Ogasawara and Fabio Porto and Eduardo Bezerra",
	title 	= "STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting",
	journal = "Neurocomputing",
	volume 	= "426",
	pages 	= "285 - 298",
	year 	= "2021",
	issn 	= "0925-2312",
	doi 	= "https://doi.org/10.1016/j.neucom.2020.09.060"
}
```

## Contact
To give your opinion about this work, send an email to `rafaela.nascimento@eic.cefet-rj.br`.
