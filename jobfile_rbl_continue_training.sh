#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J RBL_long_0

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 03:59:00

# Number of cores to be allocated (multiple of 40)
#SBATCH -N 5
#SBATCH -n 200
#SBATCH --ntasks-per-node=40

#SBATCH -e error_file_RBL_CNT.e
#SBATCH -o output_file_RBL_CNT.o

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add nest/2.2.2
module add python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

# EITHER START NEW TRAINING:
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [TRAINING_STIMULI_FILE] 0 > delme_rbl_0 2>&1

# e.g.
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV11_nX7.dat 0 > delme_rbl_0 2>&1

# OR CONTINUE TRAINING
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [OLD_FOLDER] [NEW_FOLDER] [TRAINING_STIMULI_FILE] [STIM_IDX_TO_CONTINUE] > delme_rbl_0 2>&1

# e.g.
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_withMpnNoise_titer25_nStim2_0-2_gain1.00_seeds_111_1  Training_RBL_withMpnNoise_titer25_nStim2_2-4_gain1.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 2 > delme_rbl_2 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim12_0-12_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2  Training_RBL_titer25_TRJ_CNT__nStim12_0-12_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 0 > delme_rbl_K5_-5_0 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_0-12_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_12-24_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 12 > delme_rbl_K5_-5_12 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_12-24_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_24-36_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 24 > delme_rbl_K5_-5_24 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_24-36_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_36-48_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 36 > delme_rbl_K5_-5_36 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_36-48_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_48-60_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 48 > delme_rbl_K5_-5_48 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_48-60_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_60-72_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 60 > delme_rbl_K5_-5_60 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_60-72_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_72-84_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 72 > delme_rbl_K5_-5_72 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_72-84_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_84-96_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 84 > delme_rbl_K5_-5_84 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_84-96_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_96-108_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 96 > delme_rbl_K5_-5_96 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_CNT__nStim12_96-108_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_CNT__nStim12_108-120_gainD1_0.2_D2_0.2_K5_-5_seeds_111_2/ training_stimuli_900_3step_nonshuffled.txt 108 > delme_rbl_K5_-5_108 2>&1

echo "Stopping at `date`"
