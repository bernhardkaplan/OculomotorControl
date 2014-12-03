#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J test_g02_K2

#SBATCH -t 0:25:00

# Number of cores to be allocated (multiple of 20)
#SBATCH -n 80

# Number of cores hosting OpenMP threads

#SBATCH -e error_file_testing.e
#SBATCH -o output_file_testing.o

# Run the executable named myexe 
# and write the output into my_output_file

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add nest/2.2.2
module add python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

aprun -n 80 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_testing.py Training__gaussReward_nIt30_temp1.5_nC1__nStim15_285-300_gainD1_0.2_D2_0.2_K2_-2_seeds_111_2/ > delme_testing_g2_K2 2>&1


# RBL
#aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_iteratively_reward_based.py  Training_SubOpt_2_titer25_nRF50_nV50_nStim4x400_nactions17_blurX0.05_V0.05_taup100000/ > delme_rbl_2 2>&1

echo "Stopping at `date`"

