#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J RBL_long_0

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 04:25:00

# Number of cores to be allocated (multiple of 40)
#SBATCH -N 3
#SBATCH -n 80
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

# EITHER START NEW TRAINING:
#aprun -n 80 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [TRAINING_STIMULI_FILE] 0 > delme_rbl_0 2>&1

# e.g.
#aprun -n 80 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV11_nX7.dat 0 > delme_rbl_0 2>&1

# OR CONTINUE TRAINING
#aprun -n 80 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [OLD_FOLDER] [NEW_FOLDER] [TRAINING_STIMULI_FILE] [STIM_IDX_TO_CONTINUE] > delme_rbl_0 2>&1

# e.g.
aprun -n 80 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_nStim4_0-4_gain2.00_seeds_111_1/ Training_RBL_titer25_nStim4_4-8_gain2.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 4 > delme_rbl_4 2>&1

#aprun -n 80 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_nStim4_4-8_gain2.00_seeds_111_1  Training_RBL_titer25_nStim10_8-18_gain2.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 8 > delme_rbl_8 2>&1
#aprun -n 80 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_nStim10_8-18_gain2.00_seeds_111_1/ Training_RBL_titer25_nStim10_18-28_gain2.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 18 > delme_rbl_18 2>&1
#aprun -n 80 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_nStim10_18-28_gain2.00_seeds_111_1/ Training_RBL_titer25_nStim10_28-38_gain2.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 28 > delme_rbl_28 2>&1


echo "Stopping at `date`"


