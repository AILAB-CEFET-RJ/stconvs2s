#!/bin/bash

cd ..

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

version=4

export model
export dataset
export version

dropout_list=(0.2 0.4 0.6 0.8)
counter=1

for i in ${dropout_list[@]}; do
    dropout_rate="${i}"
    export dropout_rate 
    export counter

    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=pytorch --ExecutePreprocessor.timeout=None base-"$model".ipynb --output dropout-search/"$dataset"-"$model"-step5-v"$version"-dropout-search-"$counter".ipynb > dropout-search/"$dataset"-"$model"-v"$version"-"$counter".out 2>&1
    ((counter++))

    sleep 60
done
