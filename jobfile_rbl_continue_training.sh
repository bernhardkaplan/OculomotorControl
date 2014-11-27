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
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_cycle1__nStim9_0-9_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2  Training__RBL_titer25_TRJ_CNT__nStim15_9-24_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 9 > delme_rbl_K5_-5_9 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_9-24_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_24-39_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 24 > delme_rbl_K5_-5_24 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_24-39_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_39-54_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 39 > delme_rbl_K5_-5_39 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_39-54_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_54-69_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 54 > delme_rbl_K5_-5_54 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_54-69_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_69-84_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 69 > delme_rbl_K5_-5_69 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_69-84_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_84-99_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 84 > delme_rbl_K5_-5_84 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_84-99_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_99-114_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 99 > delme_rbl_K5_-5_99 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_99-114_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_114-129_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 114 > delme_rbl_K5_-5_114 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_114-129_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_129-144_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 129 > delme_rbl_K5_-5_129 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_129-144_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_144-159_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 144 > delme_rbl_K5_-5_144 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_144-159_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_159-174_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 159 > delme_rbl_K5_-5_159 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_159-174_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_174-189_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 174 > delme_rbl_K5_-5_174 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_174-189_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_189-204_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 189 > delme_rbl_K5_-5_189 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_189-204_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_204-219_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 204 > delme_rbl_K5_-5_204 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_204-219_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_219-234_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 219 > delme_rbl_K5_-5_219 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_219-234_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_234-249_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 234 > delme_rbl_K5_-5_234 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_234-249_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_249-264_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 249 > delme_rbl_K5_-5_249 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_249-264_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_264-279_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 264 > delme_rbl_K5_-5_264 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__RBL_titer25_TRJ_CNT__nStim15_264-279_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ Training__RBL_titer25_TRJ_CNT__nStim15_279-294_gainD1_0.4_D2_0.4_K5_-5_seeds_111_2/ training_stim_params_3steps300stim.txt 279 > delme_rbl_K5_-5_279 2>&1

echo "Stopping at `date`"
