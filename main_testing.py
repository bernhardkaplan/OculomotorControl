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
from main_training import remove_files_from_folder, save_spike_trains

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


if __name__ == '__main__':

    t0 = time.time()

    GP = simulation_parameters.global_parameters()
    params = GP.params
    if params['training']:
        print 'Set training = False!'
        exit(1)

    assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
    training_folder = os.path.abspath(sys.argv[1]) # contains the EPTH and OB activity of simple patterns
    print 'Training folder:', training_folder
    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_param_tool = simulation_parameters.global_parameters(params_fn=training_params_fn)
    training_params = training_param_tool.params

    GP.params['training_params'] = training_params
    GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
    motion_params = np.loadtxt(training_params['motion_params_fn'])


    if pc_id == 0:
        remove_files_from_folder(params['spiketimes_folder_mpn'])
        remove_files_from_folder(params['input_folder_mpn'])
    
    VI = VisualInput.VisualInput(params)
    MT = MotionPrediction.MotionPrediction(params, VI, comm)
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(params, comm)
    CC = CreateConnections.CreateConnections(params)
    CC.set_pc_id(pc_id)
    CC.connect_mt_to_bg_after_training(MT, BG, training_params)

    actions = np.zeros((params['n_iterations'] + 1, 2)) # the first row gives the initial action, [0, 0] (vx, vy)
    network_states_net= np.zeros((params['n_iterations'], 4))
    training_stimuli = np.loadtxt(training_params['training_sequence_fn'])

    iteration_cnt = 0
    for i_stim in xrange(params['n_testing_stim']):
        print 'debug vi current_motion_params', VI.current_motion_params
        print 'debug vi training stimli', training_stimuli[i_stim, :]
        VI.current_motion_params = training_stimuli[i_stim, :]
        for it in xrange(params['n_iterations_per_stim']):

            if it == params['n_iterations_per_stim'] - 1:
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                # and get the state information BEFORE MPN perceives anything
                # in order to set a supervisor signal
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, action_code=actions[iteration_cnt, :])

            print 'DEBUG, comparison global iteration i_stim %d it %d iteration_cnt %d, VI: %d' % (i_stim, it, iteration_cnt, VI.iteration)
            if params['debug_mpn']:
                print 'Iteration %d: Saving spike trains...' % iteration_cnt
                save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)
            MT.update_input(stim) # run the network for some time 
            if comm != None:
                comm.barrier()
            nest.Simulate(params['t_iteration'])
            if comm != None:
                comm.barrier()

            state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)

            network_states_net[iteration_cnt, :] = state_
            print 'Iteration: %d\t%d\tState before action: ' % (iteration_cnt, pc_id), state_

            next_state = BG.get_action(state_) # BG returns the network_states_net of the next stimulus
            actions[iteration_cnt + 1, :] = next_state
            print 'Iteration: %d\t%d\tState after action: ' % (iteration_cnt, pc_id), next_state
            iteration_cnt += 1
            if comm != None:
                comm.barrier()

    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['network_states_fn'], network_states_net)
        np.savetxt(params['motion_params_fn'], VI.motion_params)
        os.system('python PlottingScripts/PlotMPNActivity.py')

    t1 = time.time() - t0
    print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

