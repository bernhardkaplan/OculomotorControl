import sys
import os
import VisualInput
import MotionPrediction
import BasalGanglia
import json
import simulation_parameters
import CreateConnections
import nest
import numpy as np
import time
import os
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



def save_spike_trains(params, iteration, stim_list, gid_list):
    n_units = len(stim_list)
    fn_base = params['input_st_fn_mpn']
    for i_ in xrange(n_units):
        if len(stim_list[i_]) > 0:
            fn = fn_base + '%d_%d.dat' % (iteration, gid_list[i_] - 1)
            np.savetxt(fn, stim_list[i_])


def remove_files_from_folder(folder):
    print 'Removing all files from folder:', folder
    path =  os.path.abspath(folder)
    cmd = 'rm  %s/*' % path
    print cmd
    os.system(cmd)


if __name__ == '__main__':

    t1 = time.time()
    if len(sys.argv) > 1: # re-run an old parameter file
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn): # go to the path containing the json object storing old parameters
            param_fn += '/Parameters/simulation_parameters.json' # hard coded subpath in ParameterContainer
        assert os.path.exists(param_fn), 'ERROR: Can not find %s - please give an existing parameter filename or folder name to re-run a simulation' % (param_fn)
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
    else: # run a simulation with parameters as set in simulation_parameters
        GP = simulation_parameters.global_parameters()
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
        params = GP.params

    if not params['training']:
        print 'Set training = True!'
        exit(1)
    

    t0 = time.time()

    VI = VisualInput.VisualInput(params)
    MT = MotionPrediction.MotionPrediction(params, VI, comm)

    if pc_id == 0:
        remove_files_from_folder(params['spiketimes_folder_mpn'])
        remove_files_from_folder(params['input_folder_mpn'])
        remove_files_from_folder(params['connections_folder'])
    
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(params, comm)
    BG.write_cell_gids_to_file()
    CC = CreateConnections.CreateConnections(params, comm)
    CC.connect_mt_to_bg(MT, BG)

    actions = np.zeros((params['n_iterations'] + 1, 2)) # the first row gives the initial action, [0, 0] (vx, vy)
    network_states_net= np.zeros((params['n_iterations'], 4))
    for iteration in xrange(params['n_iterations']):

        # integrate the real world trajectory and the eye direction and compute spike trains from that
        # and get the state information BEFORE MPN perceives anything
        # in order to set a supervisor signal
        stim, supervisor_state = VI.compute_input(MT.local_idx_exc, action_code=actions[iteration, :])

        print 'DEBUG iteration %d pc_id %d current motion params: (x,y) (u, v)' % (iteration, pc_id), VI.current_motion_params[0], VI.current_motion_params[1], VI.current_motion_params[2], VI.current_motion_params[3]
        print 'Iteration: %d\t%d\tsupervisor_state : ' % (iteration, pc_id), supervisor_state
        BG.supervised_training(supervisor_state)

        if params['debug_mpn']:
            print 'Saving spike trains...'
            save_spike_trains(params, iteration, stim, MT.local_idx_exc)

        MT.update_input(stim) # run the network for some time 
        if comm != None:
            comm.barrier()
        nest.Simulate(params['t_iteration'])
        if comm != None:
            comm.barrier()

        state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)

        print 'debug state', iteration, state_
        network_states_net[iteration, :] = state_

        print 'Iteration: %d\t%d\tState before action: ' % (iteration, pc_id), state_
        next_state = BG.get_action(state_) # BG returns the network_states_net of the next stimulus
        actions[iteration + 1, :] = next_state
        print 'Iteration: %d\t%d\tState after action: ' % (iteration, pc_id), next_state

    CC.get_weights(MT, BG)

    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['network_states_fn'], network_states_net)
        np.savetxt(params['motion_params_fn'], VI.motion_params)


    t1 = time.time() - t0
    print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

