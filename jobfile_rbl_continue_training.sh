#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J RBL_long_0

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 23:25:00

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

# EITHER START NEW TRAINING:
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [TRAINING_STIMULI_FILE] 0 > delme_rbl_0 2>&1

# e.g.
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV11_nX7.dat 0 > delme_rbl_0 2>&1

# OR CONTINUE TRAINING
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py [OLD_FOLDER] [NEW_FOLDER] [TRAINING_STIMULI_FILE] [STIM_IDX_TO_CONTINUE] > delme_rbl_0 2>&1

# e.g.
#aprun -n 120 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_withMpnNoise_titer25_nStim2_0-2_gain1.00_seeds_111_1  Training_RBL_withMpnNoise_titer25_nStim2_2-4_gain1.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 2 > delme_rbl_2 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim3_0-3_gain0.80_wD2o-10.0_K20_seeds_111_2  Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_3-13_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 3 > delme_rbl_3 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_3-13_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_13-23_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 13 > delme_rbl_13 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_13-23_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_23-33_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 23 > delme_rbl_23 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_23-33_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_33-43_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 33 > delme_rbl_33 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_33-43_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_43-53_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 43 > delme_rbl_43 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_43-53_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_53-63_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 53 > delme_rbl_53 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_53-63_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_63-73_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 63 > delme_rbl_63 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_63-73_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_73-83_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 73 > delme_rbl_73 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_73-83_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_83-93_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 83 > delme_rbl_83 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_83-93_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_93-103_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 93 > delme_rbl_93 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_93-103_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_103-113_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 103 > delme_rbl_103 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_103-113_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_113-123_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 113 > delme_rbl_113 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_113-123_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_123-133_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 123 > delme_rbl_123 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_123-133_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_133-143_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 133 > delme_rbl_133 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_133-143_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_143-153_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 143 > delme_rbl_143 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_143-153_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_153-163_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 153 > delme_rbl_153 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_153-163_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_163-173_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 163 > delme_rbl_163 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_163-173_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_173-183_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 173 > delme_rbl_173 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_173-183_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_183-193_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 183 > delme_rbl_183 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_183-193_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_193-203_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 193 > delme_rbl_193 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_193-203_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_203-213_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 203 > delme_rbl_203 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_203-213_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_213-223_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 213 > delme_rbl_213 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_213-223_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_223-233_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 223 > delme_rbl_223 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_223-233_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_233-243_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 233 > delme_rbl_233 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_233-243_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_243-253_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 243 > delme_rbl_243 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_243-253_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_253-263_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 253 > delme_rbl_253 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_253-263_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_263-273_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 263 > delme_rbl_263 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_263-273_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_273-283_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 273 > delme_rbl_273 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_273-283_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_283-293_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 283 > delme_rbl_283 2>&1
aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_283-293_gain0.80_K20_seeds_111_2/ Training_RBL_longTrainign_withMpnNoise_titer25_nStim10_293-303_gain0.80_K20_seeds_111_2/ training_stimuli_nV16_nX20_seed2.dat 293 > delme_rbl_293 2>&1

echo "Stopping at `date`"


