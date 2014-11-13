# The name of the script is neuron_job
#PBS -N RBL_

#PBS -l walltime=0:35:00
#set which time allocation should be charged for this job

#(only needed if you belong to more than one time allocation)
#PBS -A 2013-26-19

# Number of cores to be allocated is 24
#PBS -l mppwidth=96

#PBS -e error_file_python.e
#PBS -o output_file_python.o

# Change to the work directory
cd $PBS_O_WORKDIR

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

. /opt/modules/default/etc/modules.sh
module swap PrgEnv-pgi PrgEnv-gnu
module add nest
module add site-python

export PYTHONPATH=/pdc/vol/nest/2.2.2/lib64/python2.6/site-packages:/pdc/vol/python/packages/site-python-2.6/lib64/python2.6/site-packages:/cfs/klemming/nobackup/b/bkaplan/PythonPackages/lib64/python2.6/site-packages/

#aprun -n 96 python /cfs/nobackup/b/bkaplan/bcpnn-mt/NetworkSimModule.py > delme_bcpnn_output_0 2>&1
#aprun -n 48 python /cfs/nobackup/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_NoNoise_titer25_temp2.0_nStim1x1_taup50000_gain2.00_seeds_111_0 > delme_rbl_output_0 2>&1
aprun -n 96 python /cfs/milner/scratch/b/bkaplan/OculomotorControl/main_training_reward_based_new.py Training_RBL_titer25_nStim4_0-4_gain2.00_seeds_111_1/ Training_RBL_titer25_nStim4_4-8_gain2.00_seeds_111_1/ training_stimuli_nV11_nX7.dat 4 > delme_rbl_4 2>&1

echo "Stopping at `date`"

