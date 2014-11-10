import numpy as np 
import simulation_parameters
import os
import subprocess
import time


def prepare_simulation(ps, params):
    folder_name = params['folder_name']
    print 'DEBUG prepare_simulation', params['folder_name']
    ps.set_filenames(folder_name)
    ps.create_folders()
    param_fn = ps.params['params_fn_json']
    print 'Debug prepare_simulation folder written to file', ps.params['data_folder']
    print 'Ready for simulation:\n\t%s' % (param_fn)
    ps.write_parameters_to_file(fn=param_fn)
    time.sleep(1.)


def run_simulation(training_folder, test_folder, USE_MPI):
    # specify your run command (mpirun -np X, python, ...)
#    parameter_filename = params['params_fn_json']

    if USE_MPI:
        reply = subprocess.check_output(['grep', 'processor', '/proc/cpuinfo'])
        n_proc = reply.count('processor')
        print 'reply', n_proc
        run_command = 'mpirun -np %d python main_testing.py %s %s' % (n_proc, training_folder, test_folder)
    else:
        run_command = 'python main_testing.py %s %s' % (training_folder, test_folder)
    print 'Running:\n\t%s' % (run_command)
    os.system(run_command)



if __name__ == '__main__':

    try:
        from mpi4py import MPI
        USE_MPI = True
        comm = MPI.COMM_WORLD
        pc_id, n_proc = comm.rank, comm.size
        print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
    except:
        USE_MPI = False
        pc_id, n_proc, comm = 0, 1, None
        print "MPI not used"

#    USE_MPI = False

    training_params_fn = 'training_stimuli_nV11_nX7.dat'
    ps = simulation_parameters.global_parameters()
    n_jobs = 3
    n_stimuli_per_run = 4

    job_list = 
    for i_ in xrange(n_jobs):
        params = ps.params
#        params[param_name_1] = p1 
        stim_range = (i_ * n_stimuli_per_run, (i_ + 1) * n_stimuli_per_run)
        folder_name = 'Training_%s_nStim%d_%d-%d_gain%.2f_seeds_%d_%d/' % (params['sim_id'], \
                params['n_stim_training'], stim_range[0], stim_range[1], 
                params['gain_MT_d2'], params['master_seed'], params['visual_stim_seed'])
        print 'folder_name:', folder_name
#                params['mpn_d1_weight_amplification'] = params['gain_MT_d1']

#        params['folder_name'] = folder_name
        prepare_simulation(ps, params)
#        if comm != None:
#            comm.barrier()

#        run_simulation(training_folder, folder_name, USE_MPI)
#        if comm != None:
#            comm.barrier()



