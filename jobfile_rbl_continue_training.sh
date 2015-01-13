#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J 9action_newRewFct

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 23:59:00

# Number of cores to be allocated (multiple of 40)
#SBATCH -N 6
#SBATCH -n 240
#SBATCH --ntasks-per-node=40

#SBATCH -e error_file_RBL_CNT_5actions.e
#SBATCH -o output_file_RBL_CNT_5actions.o

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
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_90-105_gainD1_0.8_D2_0.8_K2_-2_seeds_201_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_105-120_gainD1_0.8_D2_0.8_K2_-2_seeds_216_4/ training_stim_params_3steps300stim.txt 105 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_105_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_105-120_gainD1_0.8_D2_0.8_K2_-2_seeds_216_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_120-135_gainD1_0.8_D2_0.8_K2_-2_seeds_231_4/ training_stim_params_3steps300stim.txt 120 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_120_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_120-135_gainD1_0.8_D2_0.8_K2_-2_seeds_231_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_135-150_gainD1_0.8_D2_0.8_K2_-2_seeds_246_4/ training_stim_params_3steps300stim.txt 135 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_135_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_135-150_gainD1_0.8_D2_0.8_K2_-2_seeds_246_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_150-165_gainD1_0.8_D2_0.8_K2_-2_seeds_261_4/ training_stim_params_3steps300stim.txt 150 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_150_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_150-165_gainD1_0.8_D2_0.8_K2_-2_seeds_261_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_165-180_gainD1_0.8_D2_0.8_K2_-2_seeds_276_4/ training_stim_params_3steps300stim.txt 165 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_165_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_165-180_gainD1_0.8_D2_0.8_K2_-2_seeds_276_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_180-195_gainD1_0.8_D2_0.8_K2_-2_seeds_291_4/ training_stim_params_3steps300stim.txt 180 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_180_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_180-195_gainD1_0.8_D2_0.8_K2_-2_seeds_291_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_195-210_gainD1_0.8_D2_0.8_K2_-2_seeds_306_4/ training_stim_params_3steps300stim.txt 195 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_195_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_195-210_gainD1_0.8_D2_0.8_K2_-2_seeds_306_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_210-225_gainD1_0.8_D2_0.8_K2_-2_seeds_321_4/ training_stim_params_3steps300stim.txt 210 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_210_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_210-225_gainD1_0.8_D2_0.8_K2_-2_seeds_321_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_225-240_gainD1_0.8_D2_0.8_K2_-2_seeds_336_4/ training_stim_params_3steps300stim.txt 225 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_225_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_225-240_gainD1_0.8_D2_0.8_K2_-2_seeds_336_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_240-255_gainD1_0.8_D2_0.8_K2_-2_seeds_351_4/ training_stim_params_3steps300stim.txt 240 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_240_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_240-255_gainD1_0.8_D2_0.8_K2_-2_seeds_351_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_255-270_gainD1_0.8_D2_0.8_K2_-2_seeds_366_4/ training_stim_params_3steps300stim.txt 255 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_255_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_255-270_gainD1_0.8_D2_0.8_K2_-2_seeds_366_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_270-285_gainD1_0.8_D2_0.8_K2_-2_seeds_381_4/ training_stim_params_3steps300stim.txt 270 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_270_9actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_270-285_gainD1_0.8_D2_0.8_K2_-2_seeds_381_4/ Training__OptRewFct_nactions9_30_temp0.5_nC1__nStim15_285-300_gainD1_0.8_D2_0.8_K2_-2_seeds_396_4/ training_stim_params_3steps300stim.txt 285 > delme_rbl_gD10.8_gD20.8K2_-2_tmp0.5_285_9actions 2>&1
echo "Stopping at `date`"


