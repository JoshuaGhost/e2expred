#!/bin/bash

#tar -xzf packages.tar.gz
#export PYTHONPATH=$PWD/packages
conda init bash
conda activate pytorch
#. ../.venv3/bin/activate
python3 ./test.py
#PYTHONPATH=$PYTHONPATH:/home/zzhang/.local/lib/python3.7/site-packages python3 ./test.py
