#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J RBL_long_0

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 23:59:00

# Number of cores to be allocated (multiple of 40)
#SBATCH -N 5
#SBATCH -n 200
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
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [TRAINING_STIMULI_FILE] 0 > delme_rbl_0 2>&1

# e.g.
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV11_nX7.dat 0 > delme_rbl_0 2>&1

# OR CONTINUE TRAINING
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [OLD_FOLDER] [NEW_FOLDER] [TRAINING_STIMULI_FILE] [STIM_IDX_TO_CONTINUE] > delme_rbl_0 2>&1

# e.g.
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_withMpnNoise_titer25_nStim2_0-2_gain1.00_seeds_111_1  Training_RBL_withMpnNoise_titer25_nStim2_2-4_gain1.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 2 > delme_rbl_2 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim12_0-12_gainD1_0.8_D2_0.8_K5_-5_seeds_111_2  Training_RBL_titer25_TRJ_small__nStim10_0-10_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 0 > delme_rbl_K5_-5_0 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_0-10_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_10-20_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 10 > delme_rbl_K5_-5_10 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_10-20_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_20-30_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 20 > delme_rbl_K5_-5_20 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_20-30_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_30-40_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 30 > delme_rbl_K5_-5_30 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_30-40_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_40-50_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 40 > delme_rbl_K5_-5_40 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_40-50_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_50-60_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 50 > delme_rbl_K5_-5_50 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_50-60_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_60-70_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 60 > delme_rbl_K5_-5_60 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_60-70_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_70-80_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 70 > delme_rbl_K5_-5_70 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_70-80_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_80-90_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 80 > delme_rbl_K5_-5_80 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_80-90_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_90-100_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 90 > delme_rbl_K5_-5_90 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_90-100_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_100-110_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 100 > delme_rbl_K5_-5_100 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_100-110_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_110-120_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 110 > delme_rbl_K5_-5_110 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_110-120_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_120-130_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 120 > delme_rbl_K5_-5_120 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_120-130_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_130-140_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 130 > delme_rbl_K5_-5_130 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_130-140_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_140-150_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 140 > delme_rbl_K5_-5_140 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_140-150_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_150-160_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 150 > delme_rbl_K5_-5_150 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_150-160_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_160-170_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 160 > delme_rbl_K5_-5_160 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_160-170_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_170-180_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 170 > delme_rbl_K5_-5_170 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_170-180_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_180-190_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 180 > delme_rbl_K5_-5_180 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_TRJ_small__nStim10_180-190_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ Training_RBL_titer25_TRJ_small__nStim10_190-200_gainD1_0.3_D2_0.3_K5_-5_seeds_111_2/ training_stimuli_1200.txt 190 > delme_rbl_K5_-5_190 2>&1

echo "Stopping at `date`"
