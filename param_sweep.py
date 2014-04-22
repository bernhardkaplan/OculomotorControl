import numpy as np
import simulation_parameters
import os


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


def run_simulation(training_folder, test_folder, USE_MPI):
    # specify your run command (mpirun -np X, python, ...)
#    parameter_filename = params['params_fn_json']
    if USE_MPI:
        run_command = 'mpirun -np 8 python main_testing.py %s %s' % (training_folder, test_folder)
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
#    training_folder = 'Training_Cluster_taup45000_nStim200_nExcMpn2400_nStates20_nActions21_it15-45000_wMPN-BG1.50_bias10.00/'
#    training_folder = 'Training_Cluster_taup90000_nStim400_nExcMpn2400_nStates20_nActions21_it15-90000_wMPN-BG1.50_bias10.00/'
    training_folder = 'Training_Show_taup100_nStim1_it20-200_wD12.5_wD210.0_bias0.10_K1.00/'

    print 'DEBUG PS INIT'
    ps = simulation_parameters.global_parameters()
#    param_range_1 = [0.5]
#    param_range_2 = [0.5]
    param_range_1 = np.arange(0.5, 5.5, 0.5)
    param_range_2 = [0.5, 1., 2., 5., 10.]

    param_name_1 = 'mpn_bg_weight_amplification'
    param_name_2 = 'mpn_bg_bias_amplification'
    for j_, p2 in enumerate(param_range_2):
        for i_, p1 in enumerate(param_range_1):
            params = ps.params
            params[param_name_1] = p1 
            params[param_name_2] = p2  
            test_folder = 'Test_%s_%d-%d' % (params['sim_id'], params['test_stim_range'][0], params['test_stim_range'][-1])
            test_folder += '_nStim%d_nExcMpn%d_nStates%d_nActions%d_it%d-%d_wMPN-BG%.2f_bias%.2f/' % \
                    (params['n_stim_training'], params['n_exc_mpn'], params['n_states'], \
                    params['n_actions'], params['t_iteration'], params['t_sim'], \
                    params['mpn_bg_weight_amplification'], params['mpn_bg_bias_amplification'])

            params['folder_name'] = test_folder 
            prepare_simulation(ps, params)
            if comm != None:
                comm.barrier()

#    for j_, p2 in enumerate(param_range_2):
#        for i_, p1 in enumerate(param_range_1):
#            params = ps.params
#            params[param_name_1] = p1 
#            params[param_name_2] = p2  
#            test_folder = 'Test_%s_%d-%d' % (params['sim_id'], params['test_stim_range'][0], params['test_stim_range'][-1])
#            test_folder += '_nStim%d_nExcMpn%d_nStates%d_nActions%d_it%d-%d_wMPN-BG%.2f_bias%.2f/' % \
#                    (params['n_stim_training'], params['n_exc_mpn'], params['n_states'], \
#                    params['n_actions'], params['t_iteration'], params['t_sim'], \
#                    params['mpn_bg_weight_amplification'], params['mpn_bg_bias_amplification'])
#            params['folder_name'] = test_folder 
#            prepare_simulation(ps, params)
            run_simulation(training_folder, test_folder, USE_MPI)
            if comm != None:
                comm.barrier()



