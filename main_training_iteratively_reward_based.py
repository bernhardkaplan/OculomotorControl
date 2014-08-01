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
import utils
import random

try: 
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI could not be loaded\nPlease install python-mpi4py"
    print 'utils.communicate_local_spikes will not work, hence action readout will give false results\nWill now quit'
    print 'If you are sure, that you want to run on a single core, remove the exit(1) statement'
#    exit(1)


def save_spike_trains(params, iteration, stim_list, gid_list):
    assert (len(stim_list) == len(gid_list))
    n_units = len(stim_list)
    fn_base = params['input_st_fn_mpn']
    for i_, nest_gid in enumerate(gid_list):
        if len(stim_list[i_]) > 0:
            fn = fn_base + '%d_%d.dat' % (iteration, nest_gid - 1)
            np.savetxt(fn, stim_list[i_])


def remove_files_from_folder(folder):
    print 'Removing all files from folder:', folder
    path =  os.path.abspath(folder)
    cmd = 'rm  %s/*' % path
    print cmd
    os.system(cmd)


if __name__ == '__main__':

    t1 = time.time()

    write_params = True
    GP = simulation_parameters.global_parameters()
    if len(sys.argv) < 3:
        params = GP.params
    else:
        testing_params_json = utils.load_params(os.path.abspath(sys.argv[2]))
        params = utils.convert_to_NEST_conform_dict(testing_params_json)
        write_params = False
    
    assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
    training_folder = os.path.abspath(sys.argv[1]) 
    print 'Training folder:', training_folder
    training_params_json = utils.load_params(training_folder)
    training_params = utils.convert_to_NEST_conform_dict(training_params_json)
    params['training_folder'] = training_folder
    if pc_id == 0 and write_params:
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation

    if comm != None:
        comm.Barrier()


    if not params['training']:
        print 'Set training = True!'
        exit(1)
    if params['reward_based_learning'] == False:
        print 'Set reward_based_learning = True'
        exit(1)
    

    t0 = time.time()

    VI = VisualInput.VisualInput(params, comm=comm)
    MT = MotionPrediction.MotionPrediction(params, VI, comm)
#    exit(1)

    if pc_id == 0:
        remove_files_from_folder(params['spiketimes_folder'])
        remove_files_from_folder(params['input_folder_mpn'])
        remove_files_from_folder(params['connections_folder'])
    
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(params, comm)
    CC = CreateConnections.CreateConnections(params, comm)
    CC.set_pc_id(pc_id)
#    CC.connect_mt_to_bg(MT, BG)
    CC.connect_mt_to_bg_after_training(MT, BG, training_params, params, model='bcpnn_synapse') # connect with zero weights via BCPNN synapses

    actions = np.zeros((params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
    network_states_net = np.zeros((params['n_iterations'], 4))
    rewards = np.zeros(params['n_iterations'])
    iteration_cnt = 0
#    training_stimuli = VI.create_training_sequence_iteratively()
#    training_stimuli = VI.create_training_sequence_from_a_grid()

    training_stimuli_sample = VI.create_training_sequence_iteratively()
    training_stimuli_grid = VI.create_training_sequence_from_a_grid()
    training_stimuli_center = VI.create_training_sequence_around_center()
    training_stimuli = np.zeros((params['n_stim_training'], 4))
    n_grid = int(np.round(params['n_stim_training'] * params['frac_training_samples_from_grid']))
    n_center = int(np.round(params['n_stim_training'] * params['frac_training_samples_center']))
    random.seed(params['visual_stim_seed'])
    np.random.seed(params['visual_stim_seed'])
    training_stimuli[:n_grid, :] = training_stimuli_grid[random.sample(range(params['n_stim_training']), n_grid), :]
    training_stimuli[n_grid:n_grid+n_center, :] = training_stimuli_center 
    training_stimuli[n_grid+n_center:, :] = training_stimuli_sample[random.sample(range(params['n_stim_training']), params['n_stim_training'] - n_grid - n_center), :]
    np.savetxt(params['training_sequence_fn'], training_stimuli)

    supervisor_states, action_indices, motion_params_precomputed = VI.get_supervisor_actions(training_stimuli, BG)
    print 'supervisor_states:', supervisor_states
    print 'action_indices:', action_indices
    np.savetxt(params['supervisor_states_fn'], supervisor_states)
    np.savetxt(params['action_indices_fn'], action_indices, fmt='%d')
    np.savetxt(params['motion_params_precomputed_fn'], motion_params_precomputed)

    v_eye = [0., 0.]
    for i_stim in xrange(params['n_stim']):
        VI.current_motion_params = training_stimuli[i_stim, :]
        for it_ in xrange(params['n_iterations_per_stim']):

            if it_ >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                # and get the state information BEFORE MPN perceives anything
                # in order to set a supervisor signal
                stim, supervisor_state = VI.compute_input_open_loop(MT.local_idx_exc)

            #print 'DEBUG iteration %d pc_id %d current motion params: (x,y) (u, v)' % (it_, pc_id), VI.current_motion_params[0], VI.current_motion_params[1], VI.current_motion_params[2], VI.current_motion_params[3]
            print 'Iteration: %d\t%d\tsupervisor_state : ' % (iteration_cnt, pc_id), supervisor_state
            if it_ >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                BG.set_empty_input()
            else:
                (action_index_x, action_index_y) = BG.softmax_action_selection(supervisor_state)
            print 'DEBUG action_index_x / y:', action_index_x, action_index_y

            if params['debug_mpn']:
                print 'Saving spike trains...'
#                save_spike_trains(params, iteration_cnt, stim, MT.exc_pop)
                save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)

#            print 'debug iteration %d stim' % (iteration_cnt), stim
            MT.update_input(stim) # run the network for some time 
            if comm != None:
                comm.Barrier()
            nest.Simulate(params['t_iteration'])
            if comm != None:
                comm.Barrier()

            state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            if it_ < 1:
                R = 0
            else:
                R = VI.get_reward_from_perceived_stim(state_)

            rewards[iteration_cnt] = R
            if it_ >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                if R >= 0:
                    BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=R, gain=0)
                    BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0., gain=0) 
                else:
                    BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0., gain=0)
                    BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=-R, gain=0)

            if pc_id == 0:
                print 'DEBUG Iteration %d\tstate ' % (iteration_cnt), state_
            network_states_net[iteration_cnt, :] = state_

            next_action = BG.get_action() # BG returns the network_states_net of the next stimulus
            print 'Iteration: %d\t%d\tState before action: ' % (iteration_cnt, pc_id), state_, '\tnext action: ', next_action
            v_eye[0] = next_action[0]
            v_eye[1] = next_action[1]
            actions[iteration_cnt + 1, :] = next_action
            #print 'Iteration: %d\t%d\tState after action: ' % (iteration_cnt, pc_id), next_action

            if params['weight_tracking']:
                CC.get_weights(MT, BG, iteration=iteration_cnt)

            iteration_cnt += 1
            if comm != None:
                comm.Barrier()


    CC.get_d1_d1_weights(BG)
    CC.get_weights(MT, BG)

    if comm != None:
        comm.Barrier()

    CC.merge_connection_files(params)

    t1 = time.time() - t0
    print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['network_states_fn'], network_states_net)
        np.savetxt(params['motion_params_fn'], VI.motion_params)
        np.savetxt(params['rewards_given_fn'], rewards)

        utils.remove_empty_files(params['connections_folder'])
        utils.remove_empty_files(params['spiketimes_folder'])
        if not params['Cluster']:
            os.system('python PlottingScripts/PlotBGActivity.py')
            os.system('python PlottingScripts/PlotMPNActivity.py')

    if comm != None:
        comm.barrier()

