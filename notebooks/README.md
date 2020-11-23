The `/notebooks` folder has jupyter notebooks with the results of the first round of experiments. We conduct experiments with distinct numbers of layers, filters, and kernel sizes to investigate the best hyperparameters to fit the deep learning models. 

## Usage

**These notebooks are parameterized**. Therefore, to run it again use the following commands:

```
cd experiments
./run-notebooks.sh

```

When you run [run-notebooks.sh](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/experiments/run-notebooks.sh) without parameter, it runs experiment experiments with default values. This script injects the [settings.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/experiments/settings.py) into the [notebook-builder.ipynb](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/experiments/notebook-builder.ipynb) and at the end generates a new notebook with the results.


### Positional Parameters

[1] `stconvs2s-r, stconvs2s-c, convlstm, predrnn, mim`: choose between these models. Default model: stconvs2s-r

[2] `cfsr chirps`: choose between these datasets. Default dataset: temperature (CFSR)


### Example

```
run-notebooks.sh stconvs2s-c chirps 
```