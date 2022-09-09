#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -m ae
#PBS -M yotsujisho@keio.jp

cd $PBS_O_WORKDIR

source py368/bin/activate
python3 run_hpc.py
