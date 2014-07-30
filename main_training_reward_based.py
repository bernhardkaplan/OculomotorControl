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
from PlottingScripts.PlotBGActivity import run_plot_bg
from PlottingScripts.PlotMPNActivity import MetaAnalysisClass

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




class RewardBasedLearning(object):

    def __init__(self, params, comm):
        self.params = params
        self.comm = comm
        if params['reward_based_learning'] == False:
            print 'Set reward_based_learning = True'
            exit(1)
        if params['training'] == False:
            print 'Set training = True'
            exit(1)
        if params['supervised_on'] == False:
            print 'Set supervised_on = True'
            exit(1)
        self.RNG = np.random
        self.RNG.seed(self.params['visual_stim_seed'])
        self.motion_params = np.zeros((self.params['n_iterations'], 5))  # + 1 dimension for the time axis
        self.stim_cnt = 0
        self.iteration_cnt = 0

    def create_networks(self):
        self.VI = VisualInput.VisualInput(self.params, comm=self.comm)
        self.MT = MotionPrediction.MotionPrediction(self.params, self.VI, self.comm)
        self.VI.set_pc_id(pc_id)
        self.BG = BasalGanglia.BasalGanglia(self.params, self.comm)
        self.CC = CreateConnections.CreateConnections(self.params, self.comm)


    def run_doing_action(self, stim_params, action, K=0, gain=1.):
        """
        """
        v_eye = [0., 0.]
        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, v_eye)
#        stim, supervisor_state = self.VI.compute_input_open_loop(self.MT.local_idx_exc)
        (action_index_x, action_index_y) = self.BG.supervised_training(action)
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        self.MT.update_input(stim) # run the network for some time 

        self.VI.current_motion_params = stim_params
        self.motion_params[self.iteration_cnt, :4] = self.VI.current_motion_params # store the current motion parameters before they get updated
        self.motion_params[self.iteration_cnt, -1] = self.VI.t_current

        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        if pc_id == 0:
            print 'DEBUG Iteration %d\tstate ' % (self.iteration_cnt), state_
        if self.stim_cnt == 0:
            self.rewards[self.iteration_cnt] = self.VI.get_reward_from_perceived_stim(state_)
        self.network_states[self.iteration_cnt, :] = state_
        self.iteration_cnt += 1




    def simulate_stimulus(self, stim_params, action):


        ######################################
        #
        #    C O N N E C T    M T ---> B G 
        #
        #######################################
        self.CC.connect_mt_to_bg(self.MT, self.BG)

        self.VI.current_motion_params = stim_params
        v_eye = [0., 0.]
        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, v_eye)
        (action_index_x, action_index_y) = self.BG.supervised_training(action)
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        self.MT.update_input(stim) 

        if comm != None:
            comm.Barrier()
        self.CC.get_weights(self.MT, self.BG, iteration=self.iteration_cnt)

        ######################################
        #
        #    R U N    0
        #
        #######################################
        nest.Simulate(params['t_iteration'])
        if comm != None:
            comm.Barrier()

        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        if pc_id == 0:
            print 'DEBUG Iteration %d\tstate ' % (self.iteration_cnt), state_
        self.network_states[self.iteration_cnt, :] = state_
        #print 'Iteration: %d\t%d\tState before action: ' % (self.iteration_cnt, pc_id), state_
        next_action = self.BG.get_action() # BG returns the network_states of the next stimulus
        v_eye[0] = next_action[0]
        v_eye[1] = next_action[1]
        self.actions_taken[self.iteration_cnt + 1, :] = next_action
        #print 'Iteration: %d\t%d\tState after action: ' % (self.iteration_cnt, pc_id), next_action

        if params['weight_tracking']:
            self.CC.get_weights(self.MT, self.BG, iteration=self.iteration_cnt)
        self.iteration_cnt += 1

        if comm != None:
            comm.Barrier()

        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, v_eye)
        self.MT.update_input(stim) 
        ######################################
        #
        #    R U N    1
        #
        #######################################
        nest.Simulate(self.params['t_iteration'])
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(params, self.iteration_cnt, stim, self.MT.local_idx_exc)

        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states[self.iteration_cnt, :] = state_
        self.rewards[self.iteration_cnt] = self.VI.get_reward_from_perceived_stim(state_)
        print 'DEBUG, rewards', self.rewards

        R = self.rewards[self.iteration_cnt]
        if R >= 0:
            self.BG.set_kappa_and_gain(self.MT.local_idx_exc, self.BG.strD1, kappa=R, gain=0)
            self.BG.set_kappa_and_gain(self.MT.local_idx_exc, self.BG.strD2, kappa=0., gain=0) 
        else:
            self.BG.set_kappa_and_gain(self.MT.local_idx_exc, self.BG.strD1, kappa=0., gain=0)
            self.BG.set_kappa_and_gain(self.MT.local_idx_exc, self.BG.strD2, kappa=-R, gain=0)

        if params['weight_tracking']:
            self.CC.get_weights(self.MT, self.BG, iteration=self.iteration_cnt)
        self.iteration_cnt += 1

        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 

        if comm != None:
            comm.Barrier()

        for i_ in xrange(self.params['n_silent_iterations']):
            ######################################
            #
            #    R U N 
            #
            #######################################
            nest.Simulate(self.params['t_iteration'])
            self.CC.get_weights(self.MT, self.BG, iteration=self.iteration_cnt)
            self.iteration_cnt += 1

        self.CC.get_weights(self.MT, self.BG)
        utils.merge_connection_files(self.params)

    def test_non_optimal_action(self, training_stimuli, i_stim=0):
        """
        Demonstrate a non-optimal action.
        Simulate for two iterations and get the reward.

        First compute the MPN->BG connections which should have positive weight in order 
        to trigger the non-optimal action.
        K = 0, gain > 0
        """

        actions_taken = np.zeros((params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
#        for i_ in xrange(training_stimuli[:, 0].size):
#            print 'i_', i_
#            print 'training_stimuli:', training_stimuli[i_, :]
#            print 'supervisor_states', supervisor_states[i_]
#            print 'action_indices', action_indices

#        plus_minus = utils.get_plus_minus(self.RNG)
        non_optimal_action = action_indices[0] - 1
        vx = self.BG.action_bins_x[non_optimal_action]
        action_ = (vx, 0, non_optimal_action)
        print 'Choosing to do:', action_, 'index:', non_optimal_action
        actions_taken[0, :] = non_optimal_action
    
        # update Visual input
        self.VI.current_motion_params = training_stimuli[i_stim, :]
        self.motion_params[self.VI.iteration, :4] = self.VI.current_motion_params # store the current motion parameters before they get updated
        self.motion_params[self.VI.iteration, -1] = self.VI.t_current
        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, action_)

        self.MT.update_input(stim) # run the network for some time 
        print 'supervisor_state:', supervisor_state
        idx = np.nonzero(np.array(stim))[0]
#        print 'debug idx:', idx, type(idx)
#        print 'debug type local_idx_exc', type(self.MT.local_idx_exc)
        if len(idx) > 0:
            print 'debug gids:', np.array(self.MT.local_idx_exc)[idx], pc_id
            active_mpn_neurons = list(np.array(self.MT.local_idx_exc)[idx])
        else:
            active_mpn_neurons = []
        print 'active_mpn_neurons:', active_mpn_neurons, type(active_mpn_neurons), pc_id

        w_dummy = 100.
        tgt_pop = self.BG.strD1[non_optimal_action]

        gain = 1.
        nest.ConvergentConnect(self.MT.exc_pop, self.BG.strD1[non_optimal_action], model=self.params['synapse_d1_MT_BG'])
        self.BG.set_kappa_and_gain(self.MT.local_idx_exc, self.BG.strD1, kappa=0., gain=gain)
        self.BG.set_kappa_and_gain(self.MT.local_idx_exc, self.BG.strD2, kappa=0., gain=gain)
        syn_params = {'p_ij' : np.exp(w_dummy / gain) * self.params['bcpnn_init_pi']**2, 'p_i': self.params['bcpnn_init_pi'], \
                'p_j': self.params['bcpnn_init_pi'], 'weight': w_dummy, 'K': 0., 'gain':gain}
        if len(active_mpn_neurons) > 0:
            conn_buffer = nest.GetConnections(active_mpn_neurons, tgt_pop)
            if conn_buffer != None:
                for c in conn_buffer:
                    cp = nest.GetStatus([c])
                    print 'cp:', cp
                    if cp[0]['synapse_model'] == 'bcpnn_synapse':
    #        print 'debug connbuffer:', conn_buffer
                        nest.SetStatus(conn_buffer, syn_params)
    #        print 'Stim:', stim
#            print ' debug', nest.GetConnections(active_mpn_neurons, tgt_pop)
#            print ' debug', nest.GetStatus(nest.GetConnections(active_mpn_neurons, tgt_pop))
        if self.comm != None:
            self.comm.Barrier()
        nest.Simulate(3 * params['t_iteration'])
#        print ' debug', nest.GetStatus(nest.GetConnections(active_mpn_neurons, tgt_pop))

#        nest.Simulate(params['t_iteration'])
#        nest.Simulate(params['t_iteration'])
#        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, actions[iteration_cnt, :])
        # choose a non-optimal action
#        all_actions = range(self.params['n_actions'])
#        all_actions.remove(action_indices[0])

    def test_optimal_action(self):
        """
        K = 1, gain = 0
        as during the normal 'open-loop' training, the optimal action to a given stimulus is selected.
        Additionally, the corresponding reward is computed and stored for later training (using the efference copy).
        """
        pass

        
    def train_efference_copy(self):#, stim_params, reward):
        """
        Repeat presentation of the given stimulus selecting the optimal action and give the reward as K
        """
        pass


    def set_up_data_structures(self):
        """
        Creates:
            - training stimulus sequence
        Create empty containers for:
            - taken actions
            - supervisor states
            - precomputed motion parameters
        """
        # data structures for recording
        self.network_states = np.zeros((params['n_iterations'], 4))  # readout from the visual layer
        self.actions_taken = np.zeros((params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
#        self.training_stimuli = RBL.VI.create_training_sequence_iteratively()
        self.training_stimuli = np.zeros((params['n_stim'], 4))
        self.training_stimuli[0, :] = [.7, .5, .5, .0]
#        self.training_stimuli = RBL.VI.create_training_sequence_RBL()
        self.supervisor_states, self.action_indices, self.motion_params_precomputed = self.VI.get_supervisor_actions(self.training_stimuli, self.BG)
        self.rewards = np.zeros(params['n_iterations'])
#        print 'self.training_stimuli:', self.training_stimuli
#        print 'self.training_stimuli.shape', self.training_stimuli.shape
        print 'self.supervisor_states', self.supervisor_states


    def save_data_structures(self):
        if pc_id == 0:
            utils.remove_empty_files(self.params['connections_folder'])
            utils.remove_empty_files(self.params['spiketimes_folder'])
            np.savetxt(self.params['supervisor_states_fn'], self.supervisor_states)
            np.savetxt(self.params['action_indices_fn'], self.action_indices, fmt='%d')
            np.savetxt(self.params['actions_taken_fn'], self.actions_taken)
            np.savetxt(self.params['motion_params_precomputed_fn'], self.motion_params_precomputed)
            np.savetxt(self.params['network_states_fn'], self.network_states)
            np.savetxt(self.params['rewards_given_fn'], self.rewards)
            np.savetxt(params['motion_params_fn'], self.VI.motion_params)
            np.savetxt(params['training_sequence_fn'], self.training_stimuli)

    if comm != None:
        comm.Barrier()



if __name__ == '__main__':

    GP = simulation_parameters.global_parameters()
    if pc_id == 0:
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
    params = GP.params

    if pc_id == 0:
        utils.remove_files_from_folder(params['spiketimes_folder'])
        utils.remove_files_from_folder(params['input_folder_mpn'])
        utils.remove_files_from_folder(params['connections_folder'])
    if comm != None:
        comm.Barrier()
    t0 = time.time()

    RBL = RewardBasedLearning(params, comm)

    RBL.create_networks()
    RBL.set_up_data_structures()

    ######################################
    #
    #    C O N N E C T    M T ---> B G 
    #
    #######################################
    RBL.CC.connect_mt_to_bg(RBL.MT, RBL.BG)

    # set the sequence of actions to be taken
#    action_sequence = np.zeros((params['n_stim'], 2))
#    action_sequence[0, :] = [10., 0.]
#    action_sequence[1, :] = [-3., 0.]
#    for i_stim in xrange(params['n_stim']):
#        stim_params = RBL.training_stimuli[i_stim, :]
#        RBL.run_doing_action(stim_params, action_sequence[i_stim, :])

        # simulate, do the given action and simulate again with the stimulus modified by the action
    stim_params = RBL.training_stimuli[0, :]
    action_v = [10., 0]

    RBL.simulate_stimulus(stim_params, action_v) # 2 x simulate in here
    RBL.save_data_structures()

#        action = RBL.supervisor_states[i_stim][0]
#        RBL.()
#        RBL.run_doing_action(stim_params, action)
#        new_stim = RBL.readout_new_stim()
#        RBL.run_doing_action(new_stim, action=None)
#        RBL.compute_kappa()
#        RBL.simulate_without_input()

    if pc_id == 0:
        run_plot_bg(params, (0, params['n_stim']))
        MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(params['n_stim'])])
        MAC = MetaAnalysisClass([params['folder_name']])
        run_plot_bg(params, None)
