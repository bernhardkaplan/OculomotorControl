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
    CC.connect_mt_to_bg_RBL(MT, BG, training_params, params, target='d1', model='bcpnn_synapse')
    CC.connect_mt_to_bg_RBL(MT, BG, training_params, params, target='d2', model='bcpnn_synapse')
    if params['connect_d1_after_training']:
        CC.connect_d1_after_training(BG, training_params, params)
    actions = np.zeros((params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
    network_states_net = np.zeros((params['n_iterations'], 4))
    rewards = np.zeros(params['n_iterations'])
    iteration_cnt = 0


#    training_stimuli = np.zeros((training_params['n_stim_training'], 4))
    training_stimuli = np.loadtxt(training_params['training_sequence_fn'])
    np.savetxt(params['training_sequence_fn'], training_stimuli)


    supervisor_states, action_indices, motion_params_precomputed = VI.get_supervisor_actions(training_stimuli, BG)
    print 'supervisor_states:', supervisor_states
    print 'action_indices:', action_indices
    np.savetxt(params['supervisor_states_fn'], supervisor_states)
    np.savetxt(params['action_indices_fn'], action_indices, fmt='%d')
    np.savetxt(params['motion_params_precomputed_fn'], motion_params_precomputed)

    v_eye = [0., 0.]

    for i_training_stim in xrange(params['n_training_cycles']): # how many of all the training samples shall be retrained
        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0., gain=params['gain'])
        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0., gain=params['gain'])
        for i_trial in xrange(params['n_training_stim_per_cycle']):
            VI.current_motion_params = training_stimuli[i_training_stim, :]
#            if i_trial < 2: # switch off plasticity
            BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0., gain=params['gain'])
            BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0., gain=params['gain'])
            for it_ in xrange(params['n_iterations_per_stim']):

                if it_ >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                    stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
                    it_0 = i_training_stim * params['n_training_stim_per_cycle']
                    it_1 = i_training_stim * params['n_training_stim_per_cycle'] + params['n_iterations_per_stim'] - params['n_silent_iterations']
                    if i_trial < 1:
                        BG.activate_efference_copy(it_0, it_1)
                    else:
                        BG.stop_efference_copy()
                else:
                    # closed-loop
                    stim, supervisor_state = VI.compute_input(MT.local_idx_exc, action_code=actions[iteration_cnt, :])
                    BG.stop_efference_copy()
                if params['debug_mpn']:
                    print 'Saving spike trains...'
                    save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)
                MT.update_input(stim) # run the network for some time 
                if comm != None:
                    comm.Barrier()

                # for i_trial >= 1: 'pre-choose' the BG activity 1

                nest.Simulate(params['t_iteration'])
                if comm != None:
                    comm.Barrier()

                state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
                if it_ < 1:
                    R = 0 # no learning during this iteration
                elif it_ >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                    # do not update the reward, but keep the same Kappa as before
                    pass
                else:
#                    R = VI.get_reward_from_perceived_stim(state_)
                    # pass the selected action to BG to retrieve the reward - the action is the vector average of the BG activity
                    R = BG.get_reward_from_action(actions[iteration_cnt, 2], VI.current_motion_params)
                rewards[iteration_cnt] = R

#                if it_ >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                if it_ >= 2:
                    if R >= 0:
                        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=R, gain=0)
                        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0., gain=0) 
                    else:
                        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0., gain=0)
                        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=-R, gain=0)

                if pc_id == 0:
                    print 'DEBUG Iteration %d\tstate ' % (iteration_cnt), state_
                network_states_net[iteration_cnt, :] = state_

                next_action = BG.get_action_spike_based_memory_based(i_trial, VI.current_motion_params) # BG returns the network_states_net of the next stimulus

                print 'Iteration: %d\t%d\tState before action: ' % (iteration_cnt, pc_id), state_, '\tnext action: ', next_action
                v_eye[0] = next_action[0]
                v_eye[1] = next_action[1]
                actions[iteration_cnt + 1, :] = next_action
                #print 'Iteration: %d\t%d\tState after action: ' % (iteration_cnt, pc_id), next_action

#                if params['weight_tracking']:
#                    CC.get_weights(MT, BG, iteration=iteration_cnt)
                if comm != None:
                    comm.Barrier()
                iteration_cnt += 1

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
        np.savetxt(params['nspikes_action_fn'], BG.activity_memory)

        utils.remove_empty_files(params['connections_folder'])
        utils.remove_empty_files(params['spiketimes_folder'])
        if not params['Cluster']:
            os.system('python PlottingScripts/PlotBGActivity.py')
            os.system('python PlottingScripts/PlotMPNActivity.py')

    if comm != None:
        comm.barrier()

