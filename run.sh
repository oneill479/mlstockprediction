#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate mlenv
sudo python spyML.py

exit 0
