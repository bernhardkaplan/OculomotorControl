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
    GP = simulation_parameters.global_parameters()
    if pc_id == 0:
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
    if comm != None:
        comm.Barrier()
    params = GP.params
    if params['reward_based_learning'] == False:
        print 'Set reward_based_learning = True'
        exit(1)

    if params['load_mpn_d1_weights'] or params['load_mpn_d2_weights']:
        assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
        training_folder = os.path.abspath(sys.argv[1]) 
        training_params = utils.load_params(training_folder)
    
    t0 = time.time()

    VI = VisualInput.VisualInput(params, comm=comm)
    MT = MotionPrediction.MotionPrediction(params, VI, comm)

    if pc_id == 0:
        remove_files_from_folder(params['spiketimes_folder'])
        remove_files_from_folder(params['input_folder_mpn'])
        remove_files_from_folder(params['connections_folder'])
    
    VI.set_pc_id(pc_id)

    BG = BasalGanglia.BasalGanglia(params, comm)
    CC = CreateConnections.CreateConnections(params, comm)
    if params['load_mpn_d1_weights']:
        CC.connect_mt_to_d1_after_training(MT, BG, training_params, params, model=params['mpn_d1_synapse_model'])
    if params['load_mpn_d2_weights']:
        CC.connect_mt_to_d2_after_training(MT, BG, training_params, params, model=params['mpn_d2_synapse_model'])

    actions = np.zeros((params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
    network_states_net = np.zeros((params['n_iterations'], 4))
    iteration_cnt = 0
    training_stimuli = VI.create_training_sequence_iteratively()
#    training_stimuli = VI.create_training_sequence_from_a_grid()

    supervisor_states, action_indices, motion_params_precomputed = VI.get_supervisor_actions(training_stimuli, BG)
    np.savetxt(params['supervisor_states_fn'], supervisor_states)
    np.savetxt(params['action_indices_fn'], action_indices, fmt='%d')
    np.savetxt(params['motion_params_precomputed_fn'], motion_params_precomputed)
    
    rewards = np.zeros(params['n_stim_training'] * params['n_iterations_per_stim'])

    for i_stim in xrange(params['n_stim_training']):
        VI.current_motion_params = training_stimuli[i_stim, :]

        # -----------------------------------
        # K = 0, gain = 1   T E S T I N G 
        # -----------------------------------
        # TODO:
        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0., gain=gain)
        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0., gain=gain)
        for it in xrange(params['n_iterations_per_stim'] / 2):
            if it >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, actions[iteration_cnt, :])
            if params['debug_mpn']:
                print 'Saving spike trains...'
                save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)
            MT.update_input(stim)
            if comm != None:
                comm.Barrier()
            nest.Simulate(params['t_iteration'])
            if comm != None:
                comm.Barrier()
            state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            rewards[iteration_cnt] = VI.get_reward_from_perceived_stim(state_)
            if pc_id == 0:
                print 'DEBUG Iteration %d\tREWARD: %.2f' % (iteration_cnt, rewards[iteration_cnt])
            network_states_net[iteration_cnt, :] = state_
            next_action = BG.get_action() # BG returns the action for the next stimulus
            actions[iteration_cnt + 1, :] = next_action
            if params['weight_tracking']:
                CC.get_weights(MT, BG, iteration=iteration_cnt)
            if comm != None:
                comm.Barrier()
            iteration_cnt += 1


#        BG.set_kappa_on(MT.local_idx_exc)
        VI.current_motion_params = training_stimuli[i_stim, :]
        # ------------------------------------------
        # K = Reward, gain = 0, + 'Efference' copy
        # ------------------------------------------
        for it in xrange(params['n_iterations_per_stim'] / 2):
            if pc_id == 0:
                print 'DEBUG in iteration %d\tsetting K=REWARD = %.2f' % (iteration_cnt, rewards[iteration_cnt - params['n_iterations_per_stim'] / 2])
            R = rewards[iteraction_cnt - params['n_iterations_per_stim'] / 2]
            if R > 0:
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=K, gain=gain)
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0, gain=gain)
            else:
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0, gain=gain)
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=K, gain=gain)
            if it >= (params['n_iterations_per_stim'] -  params['n_silent_iterations']):
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, actions[iteration_cnt, :])
#                rewards[iteration_cnt] = VI.get_reward()

            if params['debug_mpn']:
                print 'Saving spike trains...'
                save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)
            MT.update_input(stim)
            if comm != None:
                comm.Barrier()
            nest.Simulate(params['t_iteration'])
            if comm != None:
                comm.Barrier()
            state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            if pc_id == 0:
                print 'DEBUG Iteration %d\tREWARD: %.2f' % (iteration_cnt, rewards[iteration_cnt])
            network_states_net[iteration_cnt, :] = state_
            next_action = BG.get_action() # BG returns the network_states_net of the next stimulus
            actions[iteration_cnt + 1, :] = next_action
            if params['weight_tracking']:
                CC.get_weights(MT, BG, iteration=iteration_cnt)
            if comm != None:
                comm.Barrier()
            iteration_cnt += 1



    if pc_id == 0:
        np.savetxt(params['rewards_given_fn'], rewards)

    
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

        utils.remove_empty_files(params['connections_folder'])
        utils.remove_empty_files(params['spiketimes_folder'])
        if not params['Cluster']:
            os.system('python PlottingScripts/PlotBGActivity.py')
            os.system('python PlottingScripts/PlotMPNActivity.py')

    if comm != None:
        comm.barrier()

