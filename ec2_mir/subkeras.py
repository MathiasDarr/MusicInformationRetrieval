"""This script demonstrates how to activate the tensorflow_p36 environment on the deep learning AMI, and run another
python script from this. """
# !/usr/bin/env python3

import subprocess
import sys

if __name__ == '__main__':
    subprocess.run('bash -c "source activate /home/ubuntu/anaconda3/envs/tensorflow_p36 && python3 train_maestro_model.py " ', shell=True)

