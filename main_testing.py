import sys
import os
import numpy as np
import json
import time
import nest
import VisualInput
import MotionPrediction
import BasalGanglia
import simulation_parameters
import CreateConnections
import utils
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
    training_folder = os.path.abspath(sys.argv[1]) 
    print 'Training folder:', training_folder
    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_param_tool = simulation_parameters.global_parameters(params_fn=training_params_fn)
    training_params = training_param_tool.params
    actions = np.zeros((params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
    network_states_net= np.zeros((params['n_iterations'], 4))
    training_stimuli = np.zeros((training_params['n_stim_training'], 4))
    training_stimuli_= np.loadtxt(training_params['training_sequence_fn'])
    training_stimuli = training_stimuli_
    print 'debug', training_params['training_sequence_fn'], '\n', training_stimuli
    training_stimuli.reshape((training_params['n_stim_training'], 4))

    GP.params['training_params'] = training_params
    GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
    motion_params = np.loadtxt(training_params['motion_params_fn'])


    if pc_id == 0:
        remove_files_from_folder(params['spiketimes_folder'])
        remove_files_from_folder(params['input_folder_mpn'])
    
    VI = VisualInput.VisualInput(params)
    MT = MotionPrediction.MotionPrediction(params, VI, comm)
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(params, comm)
    BG.write_cell_gids_to_file()

    CC = CreateConnections.CreateConnections(params)
    CC.set_pc_id(pc_id)

    if comm != None:
        comm.barrier()
    CC.connect_mt_to_bg_after_training(MT, BG, training_params, params)
    BG.set_bias('d1')
    BG.set_bias('d2')

    iteration_cnt = 0
    v_eye = [0., 0.]
    for i_stim in xrange(params['n_stim_testing']):
        print 'debug vi current_motion_params', VI.current_motion_params
        print 'debug vi training stimli', training_stimuli.shape, '\n', training_stimuli
        if len(training_stimuli.shape) == 1:
            VI.current_motion_params = training_stimuli
        else:
            VI.current_motion_params = training_stimuli[i_stim, :]
        for it in xrange(params['n_iterations_per_stim']):

            if it == params['n_iterations_per_stim'] - 1:
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                # and get the state information BEFORE MPN perceives anything
                # in order to set a supervisor signal
#                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, action_code=actions[iteration_cnt, :])
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, actions[iteration_cnt, :], v_eye, network_states_net[iteration_cnt, :])

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

            next_action = BG.get_action() # BG returns the network_states_net of the next stimulus
            v_eye[0] += next_action[0]
            v_eye[1] += next_action[1]
            print 'debug next_action', next_action
            actions[iteration_cnt + 1, :] = next_action
            print 'Iteration: %d\t%d\tState after action: ' % (iteration_cnt, pc_id), next_action
            iteration_cnt += 1
            if comm != None:
                comm.barrier()

    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['network_states_fn'], network_states_net)
        np.savetxt(params['motion_params_fn'], VI.motion_params)
        utils.compare_actions_taken(training_params, params)
        os.system('python PlottingScripts/PlotMPNActivity.py')
        os.system('python PlottingScripts/PlotBGActivity.py')



    t1 = time.time() - t0
    print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

