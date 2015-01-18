#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J new_17actions

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

aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_0-30_gainD1_1.2_D2_1.2_K1_-1_seeds_321_5 Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_30-60_gainD1_1.2_D2_1.2_K1_-1_seeds_141_5/ training_stimuli_900_0.3center_0.7center.txt 30 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_30_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_30-60_gainD1_1.2_D2_1.2_K1_-1_seeds_141_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_60-90_gainD1_1.2_D2_1.2_K1_-1_seeds_171_5/ training_stimuli_900_0.3center_0.7center.txt 60 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_60_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_60-90_gainD1_1.2_D2_1.2_K1_-1_seeds_171_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_90-120_gainD1_1.2_D2_1.2_K1_-1_seeds_201_5/ training_stimuli_900_0.3center_0.7center.txt 90 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_90_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_90-120_gainD1_1.2_D2_1.2_K1_-1_seeds_201_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_120-150_gainD1_1.2_D2_1.2_K1_-1_seeds_231_5/ training_stimuli_900_0.3center_0.7center.txt 120 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_120_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_120-150_gainD1_1.2_D2_1.2_K1_-1_seeds_231_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_150-180_gainD1_1.2_D2_1.2_K1_-1_seeds_261_5/ training_stimuli_900_0.3center_0.7center.txt 150 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_150_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_150-180_gainD1_1.2_D2_1.2_K1_-1_seeds_261_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_180-210_gainD1_1.2_D2_1.2_K1_-1_seeds_291_5/ training_stimuli_900_0.3center_0.7center.txt 180 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_180_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_180-210_gainD1_1.2_D2_1.2_K1_-1_seeds_291_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_210-240_gainD1_1.2_D2_1.2_K1_-1_seeds_321_5/ training_stimuli_900_0.3center_0.7center.txt 210 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_210_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_210-240_gainD1_1.2_D2_1.2_K1_-1_seeds_321_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_240-270_gainD1_1.2_D2_1.2_K1_-1_seeds_351_5/ training_stimuli_900_0.3center_0.7center.txt 240 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_240_17actions 2>&1
aprun -n 240 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_240-270_gainD1_1.2_D2_1.2_K1_-1_seeds_351_5/ Training_NEW3_nactions17_30_temp0.5_nC1__nStim30_270-300_gainD1_1.2_D2_1.2_K1_-1_seeds_381_5/ training_stimuli_900_0.3center_0.7center.txt 270 > delme_rbl_gD11.2_gD21.2K1_-1_tmp0.5_270_17actions 2>&1


echo "Stopping at `date`"


