#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J test_0_15

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 0:15:00

# Number of cores to be allocated (multiple of 20)
#SBATCH -n 40

# Number of cores hosting OpenMP threads

#SBATCH -e error_file_testing.e
#SBATCH -o output_file_testing.o

# Run the executable named myexe 
# and write the output into my_output_file

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add nest
module add python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

#aprun -n 120 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_testing.py Training__taup150000_nStim2000_it15-300000_wD14.0_wD210.0_bias1.00_K1.00 > delme_testing 2>&1
#aprun -n 960 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_testing.py $1 $2 > delme_testing 2>&1

aprun -n 40 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_testing.py Training_SubOptimal_titer25_nRF50_nV50_vmin0.10_vmax12.00_0_nStim4x900_nactions17_blurX0.05_V0.05_taup225000/  > delme_testing_0_15 2>&1

echo "Stopping at `date`"

