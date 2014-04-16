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

    assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
    training_folder = os.path.abspath(sys.argv[1]) 
    print 'Training folder:', training_folder

    GP = simulation_parameters.global_parameters()
    if comm != None:
        comm.barrier()

    if len(sys.argv) < 3:
        testing_params = GP.params
    else:
        testing_params_json = utils.load_params(os.path.abspath(sys.argv[2]))
        testing_params = utils.convert_to_NEST_conform_dict(testing_params_json)

    if testing_params['training']:
        print 'Set training = False!'
        exit(1)
    GP.write_parameters_to_file(testing_params['params_fn_json'], testing_params) # write_parameters_to_file MUST be called before every simulation

    if comm != None:
        comm.barrier()

#    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_params = utils.load_params(training_folder)
    actions = np.zeros((testing_params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
    network_states_net= np.zeros((testing_params['n_iterations'], 4))
    training_stimuli = np.zeros((training_params['n_stim_training'], 4))
    training_stimuli_= np.loadtxt(training_params['training_sequence_fn'])
    training_stimuli = training_stimuli_
    training_stimuli.reshape((training_params['n_stim_training'], 4))

    training_params['training_params'] = training_params # double check
    motion_params = np.loadtxt(training_params['motion_params_fn'])

    if pc_id == 0:
        remove_files_from_folder(testing_params['spiketimes_folder'])
        remove_files_from_folder(testing_params['input_folder_mpn'])
    
    VI = VisualInput.VisualInput(testing_params)
    MT = MotionPrediction.MotionPrediction(testing_params, VI, comm)
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(testing_params, comm)
    BG.write_cell_gids_to_file()

    CC = CreateConnections.CreateConnections(testing_params)
    CC.set_pc_id(pc_id)

    if comm != None:
        comm.barrier()


    CC.connect_mt_to_bg_after_training(MT, BG, training_params, testing_params)
    if comm != None:
        comm.barrier()
    BG.set_bias('d1')
    if comm != None:
        comm.barrier()
    BG.set_bias('d2')
    if comm != None:
        comm.barrier()

    iteration_cnt = 0
    v_eye = [0., 0.]
    for i_, i_stim in enumerate(testing_params['test_stim_range']):
        if len(training_stimuli.shape) == 1:
            VI.current_motion_params = training_stimuli
        else:
            VI.current_motion_params = training_stimuli[i_stim, :]
        for it in xrange(testing_params['n_iterations_per_stim']):

            if it == testing_params['n_iterations_per_stim'] - 1:
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                # and get the state information BEFORE MPN perceives anything
                # in order to set a supervisor signal
#                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, action_code=actions[iteration_cnt, :])
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, actions[iteration_cnt, :], v_eye, network_states_net[iteration_cnt, :])

            if testing_params['debug_mpn']:
                print 'Iteration %d: Saving spike trains...' % iteration_cnt
                save_spike_trains(testing_params, iteration_cnt, stim, MT.local_idx_exc)
            MT.update_input(stim) # run the network for some time 
            if comm != None:
                comm.barrier()
            nest.Simulate(testing_params['t_iteration'])
            if comm != None:
                comm.barrier()

            state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)

            network_states_net[iteration_cnt, :] = state_
            print 'Iteration: %d\t%d\tState before action: ' % (iteration_cnt, pc_id), state_

            next_action = BG.get_action() # BG returns the network_states_net of the next stimulus
            v_eye[0] += next_action[0]
            v_eye[1] += next_action[1]
            actions[iteration_cnt + 1, :] = next_action
            print 'Iteration: %d\t%d\tState after action: ' % (iteration_cnt, pc_id), next_action
            iteration_cnt += 1
            if comm != None:
                comm.barrier()

    if pc_id == 0:
        np.savetxt(testing_params['actions_taken_fn'], actions)
        np.savetxt(testing_params['network_states_fn'], network_states_net)
        np.savetxt(testing_params['motion_params_fn'], VI.motion_params)
        utils.remove_empty_files(params['connections_folder'])
        utils.remove_empty_files(params['spiketimes_folder'])
        utils.compare_actions_taken(training_params, testing_params)
        if not testing_params['Cluster'] and not testing_params['Cluster_Milner']:
            os.system('python PlottingScripts/PlotMPNActivity.py %s' % testing_params['folder_name'])
            os.system('python PlottingScripts/PlotBGActivity.py %s'% testing_params['folder_name'])
            os.system('python PlottingScripts/compare_test_and_training_performance.py %s %s' % (training_params['folder_name'], testing_params['folder_name']))
    if comm != None:
        comm.barrier()

    t1 = time.time() - t0
    print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

