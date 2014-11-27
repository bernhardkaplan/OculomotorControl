#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J K5_g0.4_RB

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 01:25:00

# Number of cores to be allocated (multiple of 40)
#SBATCH -N 5
#SBATCH -n 200
#SBATCH --ntasks-per-node=40

#SBATCH -e error_file_RBL_K5.e
#SBATCH -o output_file_RBL_K5.o

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add nest/2.2.2
module add python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

aprun -n 200 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py  training_stim_params_3steps300stim.txt  0 > delme_K5_gain0.4 2>&1

#aprun -n 160 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV10_nX10_trajectory_8steps.dat  0 > delme_rbl_K20_-20 2>&1
#aprun -n 160 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV1_nX1_3cycles_5step_trajectory.txt  0 > delme_rbl_K5_-20 2>&1
#aprun -n 200 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py  training_stimuli_nV1_nX1_2cycles_4step_trajectory.txt 0 > delme_rbl_K5_5 2>&1
#aprun -n 200 -N 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py training_stimuli_nV1_nX1_3cycles_4step_trajectory.txt 0 > delme_K3_gain0.2 2>&1


#aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py > delme_rbl_2 2>&1

echo "Stopping at `date`"


