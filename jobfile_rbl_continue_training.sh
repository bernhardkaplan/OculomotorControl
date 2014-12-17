#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J LT_g4_K2_tmp1.0

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
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_0-15_gainD1_0.8_D2_0.8_K2_-2_seeds_111_2  Training__nactions13_30_temp0.5_nC1__nStim15_15-30_gainD1_0.8_D2_0.8_K2_-2_seeds_126_2/ training_stim_params_3steps300stim.txt 15 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_15 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_15-30_gainD1_0.8_D2_0.8_K2_-2_seeds_126_2/ Training__nactions13_30_temp0.5_nC1__nStim15_30-45_gainD1_0.8_D2_0.8_K2_-2_seeds_141_2/ training_stim_params_3steps300stim.txt 30 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_30 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_30-45_gainD1_0.8_D2_0.8_K2_-2_seeds_141_2/ Training__nactions13_30_temp0.5_nC1__nStim15_45-60_gainD1_0.8_D2_0.8_K2_-2_seeds_156_2/ training_stim_params_3steps300stim.txt 45 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_45 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_45-60_gainD1_0.8_D2_0.8_K2_-2_seeds_156_2/ Training__nactions13_30_temp0.5_nC1__nStim15_60-75_gainD1_0.8_D2_0.8_K2_-2_seeds_171_2/ training_stim_params_3steps300stim.txt 60 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_60 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_60-75_gainD1_0.8_D2_0.8_K2_-2_seeds_171_2/ Training__nactions13_30_temp0.5_nC1__nStim15_75-90_gainD1_0.8_D2_0.8_K2_-2_seeds_186_2/ training_stim_params_3steps300stim.txt 75 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_75 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_75-90_gainD1_0.8_D2_0.8_K2_-2_seeds_186_2/ Training__nactions13_30_temp0.5_nC1__nStim15_90-105_gainD1_0.8_D2_0.8_K2_-2_seeds_201_2/ training_stim_params_3steps300stim.txt 90 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_90 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_90-105_gainD1_0.8_D2_0.8_K2_-2_seeds_201_2/ Training__nactions13_30_temp0.5_nC1__nStim15_105-120_gainD1_0.8_D2_0.8_K2_-2_seeds_216_2/ training_stim_params_3steps300stim.txt 105 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_105 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_105-120_gainD1_0.8_D2_0.8_K2_-2_seeds_216_2/ Training__nactions13_30_temp0.5_nC1__nStim15_120-135_gainD1_0.8_D2_0.8_K2_-2_seeds_231_2/ training_stim_params_3steps300stim.txt 120 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_120 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_120-135_gainD1_0.8_D2_0.8_K2_-2_seeds_231_2/ Training__nactions13_30_temp0.5_nC1__nStim15_135-150_gainD1_0.8_D2_0.8_K2_-2_seeds_246_2/ training_stim_params_3steps300stim.txt 135 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_135 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_135-150_gainD1_0.8_D2_0.8_K2_-2_seeds_246_2/ Training__nactions13_30_temp0.5_nC1__nStim15_150-165_gainD1_0.8_D2_0.8_K2_-2_seeds_261_2/ training_stim_params_3steps300stim.txt 150 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_150 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_150-165_gainD1_0.8_D2_0.8_K2_-2_seeds_261_2/ Training__nactions13_30_temp0.5_nC1__nStim15_165-180_gainD1_0.8_D2_0.8_K2_-2_seeds_276_2/ training_stim_params_3steps300stim.txt 165 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_165 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_165-180_gainD1_0.8_D2_0.8_K2_-2_seeds_276_2/ Training__nactions13_30_temp0.5_nC1__nStim15_180-195_gainD1_0.8_D2_0.8_K2_-2_seeds_291_2/ training_stim_params_3steps300stim.txt 180 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_180 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_180-195_gainD1_0.8_D2_0.8_K2_-2_seeds_291_2/ Training__nactions13_30_temp0.5_nC1__nStim15_195-210_gainD1_0.8_D2_0.8_K2_-2_seeds_306_2/ training_stim_params_3steps300stim.txt 195 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_195 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_195-210_gainD1_0.8_D2_0.8_K2_-2_seeds_306_2/ Training__nactions13_30_temp0.5_nC1__nStim15_210-225_gainD1_0.8_D2_0.8_K2_-2_seeds_321_2/ training_stim_params_3steps300stim.txt 210 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_210 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_210-225_gainD1_0.8_D2_0.8_K2_-2_seeds_321_2/ Training__nactions13_30_temp0.5_nC1__nStim15_225-240_gainD1_0.8_D2_0.8_K2_-2_seeds_336_2/ training_stim_params_3steps300stim.txt 225 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_225 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_225-240_gainD1_0.8_D2_0.8_K2_-2_seeds_336_2/ Training__nactions13_30_temp0.5_nC1__nStim15_240-255_gainD1_0.8_D2_0.8_K2_-2_seeds_351_2/ training_stim_params_3steps300stim.txt 240 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_240 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_240-255_gainD1_0.8_D2_0.8_K2_-2_seeds_351_2/ Training__nactions13_30_temp0.5_nC1__nStim15_255-270_gainD1_0.8_D2_0.8_K2_-2_seeds_366_2/ training_stim_params_3steps300stim.txt 255 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_255 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_255-270_gainD1_0.8_D2_0.8_K2_-2_seeds_366_2/ Training__nactions13_30_temp0.5_nC1__nStim15_270-285_gainD1_0.8_D2_0.8_K2_-2_seeds_381_2/ training_stim_params_3steps300stim.txt 270 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_270 2>&1
aprun -n 200 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__nactions13_30_temp0.5_nC1__nStim15_270-285_gainD1_0.8_D2_0.8_K2_-2_seeds_381_2/ Training__nactions13_30_temp0.5_nC1__nStim15_285-300_gainD1_0.8_D2_0.8_K2_-2_seeds_396_2/ training_stim_params_3steps300stim.txt 285 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_285 2>&1

echo "Stopping at `date`"

