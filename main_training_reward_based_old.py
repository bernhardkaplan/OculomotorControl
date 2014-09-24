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
        self.VI.current_motion_params = stim_params
        stim, supervisor_state = self.VI.compute_input_open_loop(self.MT.local_idx_exc)
        self.MT.update_input(stim) # run the network for some time 
        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        if pc_id == 0:
            print 'DEBUG Iteration %d\tstate ' % (self.iteration_cnt), state_
        if self.stim_cnt == 0:
            self.rewards[self.iteration_cnt] = self.VI.get_reward_from_perceived_stim(state_)
        self.network_states[self.iteration_cnt, :] = state_
        action_ = [action, 0]
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

    if comm != None:
        comm.Barrier()



if __name__ == '__main__':


    # during development

#    folder = 'Test_RBL_titer_test30_0-2_nStim2x400_wampD10.3_wampD20.3_d1d1wap0.00e+00_d1d1wan3.00e+01/'
#    params_json = utils.load_params(folder)
#    params = utils.convert_to_NEST_conform_dict(params_json)

#    params['n_stim_training'] = 2

#    t1 = time.time()

    GP = simulation_parameters.global_parameters()
    if pc_id == 0:
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
    params = GP.params

#    if pc_id == 0:
#        utils.remove_files_from_folder(params['spiketimes_folder'])
#        utils.remove_files_from_folder(params['input_folder_mpn'])
#        utils.remove_files_from_folder(params['connections_folder'])
    if comm != None:
        comm.Barrier()
    t0 = time.time()

    RBL = RewardBasedLearning(params, comm)
    # optionally load trained weights
#    if params['load_mpn_d1_weights'] or params['load_mpn_d2_weights']:
#        assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
#        training_folder = os.path.abspath(sys.argv[1]) 
#        training_params = utils.load_params(training_folder)
    RBL.create_networks()
    RBL.set_up_data_structures()

    action_sequence = np.zeros((params['n_stim'], 2))
    action_sequence[0, :] = [10., 0.]
    action_sequence[1, :] = [-3., 0.]
    for i_stim in xrange(params['n_stim']):
        stim_params = RBL.training_stimuli[i_stim, :]
#        action_v = [10., 0.]
        RBL.run_doing_action(stim_params, action_sequence[i_stim, :])

        # simulate, do the given action and simulate again with the stimulus modified by the action
#        RBL.simulate_stimulus(stim_params, action_v) # 2 x simulate in here

    RBL.save_data_structures()

#        action = RBL.supervisor_states[i_stim][0]
#        RBL.()
#        RBL.run_doing_action(stim_params, action)
#        new_stim = RBL.readout_new_stim()
#        RBL.run_doing_action(new_stim, action=None)

#        RBL.compute_kappa()
#        RBL.simulate_without_input()

    if pc_id == 0:
        run_plot_bg(params, (params['test_stim_range'][0], params['test_stim_range'][-1]))
        if params['n_stim'] > 1:
            MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(params['n_stim'])])
        else:
            MAC = MetaAnalysisClass([params['folder_name']])

#    exit(1)
#        np.savetxt(params['motion_params_precomputed_fn'], motion_params_precomputed)
    

#    RBL.test_non_optimal_action(training_stimuli[:, 0])
#    RBL.test_optimal_action()
#    RBL.train_efference_copy()


#    CC.connect_mt_to_bg(MT, BG)
#    if params['load_mpn_d1_weights']:
#        CC.connect_mt_to_d1_after_training(MT, BG, training_params, params, model=params['mpn_d1_synapse_model'])
#    if params['load_mpn_d2_weights']:
#        CC.connect_mt_to_d2_after_training(MT, BG, training_params, params, model=params['mpn_d2_synapse_model'])
#    CC.connect_mt_to_bg_random(MT, BG.strD1, params)
#    CC.connect_mt_to_bg_random(MT, BG.strD2, params)

#    iteration_cnt = 0
#    training_stimuli = VI.create_training_sequence_from_a_grid()

#    exit(1)


    """
    gain = 1.
    for i_stim in xrange(params['n_stim_training']):
        VI.current_motion_params = training_stimuli[i_stim, :]
        # -----------------------------------
        # K = 0, gain = 1   T E S T I N G 
        # During reward based learning the supervisor first chooses a non-optimal action
        # and then for a later stimulus, chooses the optimal action in order to show re-learning
        # -----------------------------------
        # TODO:
        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0., gain=gain)
        BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0., gain=gain)
        for it in xrange(params['n_iterations_per_stim'] / 2):
            if it >= (params['n_iterations_per_stim'] / 2 -  params['n_silent_iterations']):
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, actions[iteration_cnt, :])
            if params['debug_mpn']:
                print 'Saving spike trains...'
                utils.save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)
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
            R = rewards[iteration_cnt - params['n_iterations_per_stim'] / 2]
            if R >= 0:
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=R, gain=0)
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=0., gain=0) 
            else:
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD1, kappa=0., gain=0)
                BG.set_kappa_and_gain(MT.local_idx_exc, BG.strD2, kappa=-R, gain=0)

            ### EFFERENCE COPY STUFF
            if it >= (params['n_iterations_per_stim'] / 2 -  params['n_silent_iterations']):
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, actions[iteration_cnt, :])
#                rewards[iteration_cnt] = VI.get_reward()

            if params['debug_mpn']:
                print 'Saving spike trains...'
                utils.save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)
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

        utils.remove_empty_files(params['connections_folder'])
        utils.remove_empty_files(params['spiketimes_folder'])
        if not params['Cluster']:
            os.system('python PlottingScripts/PlotBGActivity.py')
            os.system('python PlottingScripts/PlotMPNActivity.py')

    if comm != None:
        comm.barrier()

    """
