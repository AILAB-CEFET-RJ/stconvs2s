#!/bin/bash

cd ..

model_name=$1
dataset=$2
cuda=$3
version=4
only_training=true

models=("stconvs2s convlstm predrnn mim")
datasets=("cfsr chirps")
dropout_list=(0.2 0.5 0.8)

if [ -z "$model_name" ]; then
  model_name="stconvs2s"
fi

if [[ ! " ${models[@]} " =~ "$model_name" ]]; then
    echo "ERROR: $model_name isn't a valid model name. Choose between: $models"
    exit
fi

if [ -z "$dataset" ]; then
  dataset="cfsr"
fi

if [[ ! " ${datasets[@]} " =~ "$dataset" ]]; then
    echo "ERROR: $dataset isn't a valid model name. Choose between: $datasets"
    exit
fi

if [ "$cuda" == "cuda:1" ]; then
  cuda="1"  
else
  cuda="0"
fi

if [[ ! -e dropout-search/"$model_name" ]]; then
  mkdir -p dropout-search/"$model_name"
fi

export model_name
export dataset
export cuda
export version
export only_training

counter=1
for i in "${dropout_list[@]}"; do
    dropout_rate="${i}"
    export dropout_rate 

    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=pytorch --ExecutePreprocessor.timeout=None notebook-builder.ipynb --output dropout-search/"$model_name"/"$counter"-"$dataset"-step5-v"$version".ipynb > dropout-search/"$model_name"/"$counter"-"$dataset"-v"$version".out 2>&1
    ((counter++))

    sleep 60
done
