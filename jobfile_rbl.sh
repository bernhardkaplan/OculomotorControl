#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J RBL_K_5_-5

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 01:25:00

# Number of cores to be allocated (multiple of 40)
#SBATCH -N 3
#SBATCH -n 120
#SBATCH --ntasks-per-node=40

#SBATCH -e error_file_RBL.e
#SBATCH -o output_file_RBL.o

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add nest/2.2.2
module add python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV16_nX20_seed2_centered_first.dat  0 > delme_rbl_5_-5 2>&1
#aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py > delme_rbl_2 2>&1

echo "Stopping at `date`"


