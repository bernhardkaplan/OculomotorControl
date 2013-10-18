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
    t0 = time.time()

    VI = VisualInput.VisualInput(params)
    MT = MotionPrediction.MotionPrediction(params, VI, comm)

    if pc_id == 0:
        remove_files_from_folder(params['spiketimes_folder_mpn'])
        remove_files_from_folder(params['input_folder_mpn'])
    
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(params)
    CC = CreateConnections.CreateConnections(params)
    CC.connect_mt_to_bg(MT, BG)

    actions = np.zeros((params['n_iterations'] + 1, 2)) # the first row gives the initial action, [0, 0] (vx, vy)
    network_states_net= np.zeros((params['n_iterations'], 4))
    for iteration in xrange(params['n_iterations']):

        # integrate the real world trajectory and the eye direction and compute spike trains from that
        stim = VI.compute_input(MT.local_idx_exc, action_code=actions[iteration, :])

        # get the state information BEFORE MPN perceives anything
        # in order to set a supervisor signal
        supervisor_state =  (VI.trajectories[-1][0], VI.trajectories[-1][1], VI.current_motion_params[2], VI.current_motion_params[3])
        print 'DEBUG next stim pos: (x,y) (u, v)', VI.current_motion_params[0], VI.current_motion_params[1], VI.current_motion_params[2], VI.current_motion_params[3]

        if params['debug_mpn']:
            print 'debug stim', pc_id, len(stim), MT.local_idx_exc
            save_spike_trains(params, iteration, stim, MT.local_idx_exc)

        # compute BG input (for supervised learning)
#        target_action = VI.transform_trajectory_to_action()
        # BG.update_input(stim) #--> updates the Poisson-populations coding for the state
        # BG.train_action_output(target_action)

        # remove MT.update_input etc
        MT.update_input(stim) # run the network for some time 
        nest.Simulate(params['t_iteration'])
        if comm != None:
            comm.barrier()

        state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)

#        BG.update_poisson_layer(state_)
        network_states_net[iteration, :] = state_
        print 'Iteration: %d\t%d\tState before action: ' % (iteration, pc_id), state_
        next_state = BG.get_action(state_) # BG returns the network_states_net of the next stimulus
        actions[iteration + 1, :] = next_state
        print 'Iteration: %d\t%d\tState after action: ' % (iteration, pc_id), next_state
#        VI.update_retina_image(BG.get_eye_direction())

    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['network_states_fn'], network_states_net)
        np.savetxt(params['motion_params_fn'], VI.motion_params)


    t1 = time.time() - t0
    print 'Time: %.2f [sec]' % t1

