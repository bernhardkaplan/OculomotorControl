import numpy as np
import simulation_parameters
import os
import subprocess
import time


def prepare_simulation(ps, params):
    folder_name = params['folder_name']
    print 'debug.prepare_simulation', params['folder_name']
    print 'DEBUG PS set_filenames'
    ps.set_filenames(folder_name)
    ps.create_folders()
    param_fn = ps.params['params_fn_json']
#    print 'Writing parameters to: %s' % param_fn
    print 'Debug.prepare_simulation folder written to file', ps.params['data_folder']
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
    training_folder = 'Training_RBL_2_titer25_2500_2_nStim2x54_taup50000_gain1.00'

    ps = simulation_parameters.global_parameters()
    param_range_1 = [1.0, 1.5, 2.0, 2.5, 3.0]
    param_range_2 = [1.0, 1.5, 2.0, 2.5, 3.0]
    param_range_3 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    param_range_4 = [0.]
#    param_range_4 = [0.0001, 0.1, 0.5, 1., 10.]

    param_name_1 = 'gain_MT_d1'
    param_name_2 = 'gain_MT_d2'
    param_name_3 = 'gain_d1_d1'
    param_name_4 = 'gain_d2_d2'
#    param_name_1 = 'mpn_bg_bias_amplification'
    for l_, p4 in enumerate(param_range_4):
        for k_, p3 in enumerate(param_range_3):
            for i_, p1 in enumerate(param_range_1):
                for j_, p2 in enumerate(param_range_2):
                    params = ps.params
                    params[param_name_1] = p1 
                    params[param_name_2] = p2  
                    params[param_name_3] = p3
                    params[param_name_4] = p4  
                    folder_name = 'Test_%s_%d-%d' % (params['sim_id'], params['test_stim_range'][0], params['test_stim_range'][-1])
                    folder_name += '_nStim%dx%d_wampD1%.1f_wampD2%.1f_d1d1wap%.2e_d1d1wan%.2e/' % \
                            (params['n_training_cycles'], params['n_training_stim_per_cycle'], \
                             params['gain_MT_d1'], params['gain_MT_d2'], \
                             params['gain_d1_d1'], params['gain_d1_d1'])
                    params['mpn_d1_weight_amplification'] = params['gain_MT_d1']
                    params['mpn_d2_weight_amplification'] = params['gain_MT_d2']
                    params['mpn_bg_bias_amplification_d1'] = params['bias_gain']
                    params['mpn_bg_bias_amplification_d2'] = params['bias_gain']
                    params['d1_d1_weight_amplification_neg'] = params['gain_d1_d1']
                    params['d1_d1_weight_amplification_pos'] = params['gain_d1_d1']

                    params['folder_name'] = folder_name
                    prepare_simulation(ps, params)
                    if comm != None:
                        comm.barrier()

                    run_simulation(training_folder, folder_name, USE_MPI)
                    if comm != None:
                        comm.barrier()



