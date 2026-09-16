#!/bin/bash

# Disable CPU optimizations to prevent crashes
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0

# Use the new Python 3.11
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 activate_clipboard_mac.py