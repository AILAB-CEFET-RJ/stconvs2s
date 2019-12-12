#!/bin/bash

if [ "$1" == "--stconvs2s" ] || [ -z "$1" ]; then
    model="stconvs2s"
fi

if [ "$1" == "--convlstm" ]; then
    model="convlstm"
fi

if [ "$2" == "--cfsr" ] || [ -z "$2" ]; then
    dataset="cfsr"
fi

if [ "$2" == "--chirps" ]; then
    dataset="chirps"
fi


export model
export dataset

for i in $(seq 1 4); do
  version="$i"
  export version
   
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=pytorch --ExecutePreprocessor.timeout=None base-"$model".ipynb --output "$dataset"-"$model"-step5-v"$i".ipynb > "$dataset"-"$model"-step5-v"$i".out 2>&1

  sleep 60
done