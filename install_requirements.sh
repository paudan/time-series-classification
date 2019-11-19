#!/usr/bin/bash

python3 -m pip install cython
python3 -m pip install -r requirements.txt
git clone https://github.com/chlubba/catch22.git
cd catch22/wrap_Python
export CFLAGS='-std=c99'
python3 setup_P3.py build
python3 setup_P3.py install
cd ../..