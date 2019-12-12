#!/bin/bash
cd ~

# Installing Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
chmod 755 Anaconda3-5.3.1-Linux-x86_64.sh
bash Anaconda3-5.3.1-Linux-x86_64.sh -b
rm Anaconda3-5.3.1-Linux-x86_64.sh

# Set path to conda
echo ". ~/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
echo "alias ipython='python -m IPython'" >> ~/.bashrc
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh

# Creating environment
conda env create -f environment-freeze.yml
conda activate pytorch

# Instaling environment as kernel
ipython kernel install --user --name pytorch
