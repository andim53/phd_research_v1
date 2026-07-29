#!/bin/sh

##PJM -L rscgrp=a-pj24001864
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=24
#PJM --mpi proc=24
#PJM -L elapse=120:00:00
#PJM -j
#PJM -X

source ~/.bashrc
conda activate gpaw_env
module load intel
module load impi

##Batch
##pjsub
##pjstat
##pjdel
##pjshowrsc --rg

##.bashrc
##module load intel
##module load impi
##PATH="$HOME/bin:$HOME/FLAPW/bin:$PATH"

##mpiexec python ./main.py
python ./main.py