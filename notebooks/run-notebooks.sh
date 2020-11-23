#!/bin/bash

model_name=$1
dataset=$2
cuda=$3

models=("stconvs2s-r stconvs2s-c convlstm predrnn mim")
datasets=("cfsr chirps")

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

if [[ ! -e "$model_name" ]]; then
  mkdir "$model_name"
fi

export model_name
export dataset
export cuda

for i in $(seq 1 4); do
  version="$i"
  export version
   
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=pytorch --ExecutePreprocessor.timeout=None notebook-builder.ipynb --output "$model_name"/"$dataset"-step5-v"$i".ipynb > "$model_name"/"$dataset"-step5-v"$i".out 2>&1

  sleep 60
done