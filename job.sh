#!/bin/bash
#PBS -l nodes=1:ppn=20

cd $PBS_O_WORKDIR

source py368/bin/activate
python3 run_hpc.py
