# Installation (Ubuntu 18.04)
## Clone & Install OpenCv
- `git clone https://github.com/primus852/mtt-models`
- `cd mtt-models`
- `sudo apt install python3-opencv`

## (optional) Install Virtual Env
- `sudo apt-get install python3-venv`
- `python3 -m venv mtt-models-env`
- `source mtt-models-env/bin/activate`

## (optional) Upgrade pip
- `pip3 install --upgrade pip`
- `sudo python3 -m pip install -U setuptools`

## Install dependencies
- `pip3 install -r requirements.txt`

## Train the Model "LongList" only
- `python3 main.py --long-epochs 2 --skip-short`

## Train the Model "ShortList" only
- `python3 main.py --short-epochs 10 --skip-long`

## Run the OpenCV Detection (WIP)
- `python3 main.py --production`

# Description
This is a standalone Script for 3 things:
1. Train a long list of Models with a fraction of the Dataset to measure performance
2. Train a Top X short list with the full dataset and out put a model to feed to OpenCV
3. Run OpenCV with one of the trained models

# First run
The first run may take some time as it will
- Download the Dataset
- Extract the downloaded zip
- Create all needed folders
