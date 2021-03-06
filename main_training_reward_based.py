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
from copy import deepcopy
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
        if comm != None:
            self.pc_id = comm.rank
            self.n_proc = comm.size
        else:
            self.pc_id = 0
            self.n_proc = 1
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
        
    def set_connection_module(self, CM):
        self.CM = CM

    def create_networks(self):
        self.VI = VisualInput.VisualInput(self.params, comm=self.comm)
        self.MT = MotionPrediction.MotionPrediction(self.params, self.VI, self.comm)
        self.BG = BasalGanglia.BasalGanglia(self.params, self.comm)
        self.CC = CreateConnections.CreateConnections(self.params, self.comm)

    def get_random_action(self):
        """
        When no activity is seen in the BG action layer, a 'random' action can be given as 'random' response.
        This function keeps track of the chosen actions and the 'random' actions to be taken.
        """
        # either choose the optimal action
        # or choose a random action, which is not in self.retrained_actions
        pass



    def train_doing_action_with_supervisor(self, stim_params, action_v, v_eye=[0., 0.]):
        """
        Simulate two iterations with one pre-determined action in between:
            1) Stimulus with stim_params is perceived and action is done as reaction to that
            2) A new stimulus with modified stimulus parameters is perceived and simulated
        Arguments:
            stim_params     --  (x, y, v_x, v_y)
            action          --  (v_x, v_y) = the action to be done represented in speed "units"
            v_eye           --  the initial speed of the eye (influences the perceived stimulus speed in the first iteration)
        """

#        print 'DEBUG, iteration %d VI.t_current ' % self.iteration_cnt, self.VI.t_current, 'current_motion_params', self.VI.current_motion_params

        #####
        # 1 # 
        #####
        # present a stimulus (with a possible 'ongoing' eye movement of v_eye)
        self.VI.current_motion_params = deepcopy(stim_params)
        self.motion_params[self.iteration_cnt, :4] = deepcopy(self.VI.current_motion_params) # store the current motion parameters before they get updated
        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, v_eye)
        # and activate the supervisor in the BG for a given action 
        # NOTE: this does not mean that the system actually performs this action, 
        # because the BG action-decision is based on the supervisor activity and the 
        # inherent activity forwarded from D1 and D2 driven by the stimulus
        # as long as BG.supervised_training is called, other actions != the chosen action will have zero activity
        (action_index_x, action_index_y) = self.BG.supervised_training(action_v)
        self.action_indices.append(action_index_x)
        print 'DEBUG train_doing_action_with_supervisor: action=', action_v, ' has been mapped to action_idx:', action_index_x, ' self.motion_params[%d, :]' % (self.iteration_cnt), self.motion_params[self.iteration_cnt, :]
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        self.MT.update_input(stim) # run the network for some time 
        # for the first two iterations switch off plasticity and set gain to 0 -- because you want to avoid triggering the wrong action in D1/D2
        # and want to have silence in D1 / D2 populations (except for the activity triggered by the supervisor)
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., 0., 0.)
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., 0., 0.)
        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states[self.iteration_cnt, :] = state_
        next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        self.actions_taken[self.iteration_cnt, :] = next_action
        self.MT.advance_iteration()
        R = self.BG.get_reward_from_action(action_index_x, self.motion_params[self.iteration_cnt, :4], training=True)
        self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
        self.iteration_cnt += 1
        #print 'DEBUG R = ', R

        #####
        # 2 #   E M P T Y    I N P U T 
        #####
        self.BG.stop_efference_copy()
        self.BG.stop_supervisor()
        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states[self.iteration_cnt, :] = state_
        next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        self.actions_taken[self.iteration_cnt, :] = next_action
        self.MT.advance_iteration()
        self.rewards[self.iteration_cnt + 1] = 0 # + 1 because the reward will affect the next iteration
        self.iteration_cnt += 1


        #####
        # 3 #   L E A R N I N G    P H A S E 
        #####
        # Now, switch on kappa, set gain to zero
        if R >= 0:
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, R, 0., 0.) # kappa, syn_gain, bias_gain
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., 0., 0.) 
        else:
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., 0., 0.)
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, -R, 0., 0.)
        self.BG.activate_efference_copy(self.iteration_cnt - 2, self.iteration_cnt - 1)
        self.BG.stop_supervisor()
        for i_ in xrange(self.params['n_iterations_RBL_retraining']):
            stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
            self.MT.update_input(stim) 
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states[self.iteration_cnt, :] = state_
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            self.actions_taken[self.iteration_cnt, :] = next_action
            # don't update R, because the plasticity should act based on the initial movement
            self.rewards[self.iteration_cnt] = R
            self.MT.advance_iteration()
            self.iteration_cnt += 1
        #self.CM.get_weights(self.MT, self.BG, iteration=self.iteration_cnt)

        #####
        # 4 #       E M P T Y    R U N 
        #####
        self.BG.stop_efference_copy()
        self.BG.stop_supervisor()
        for it_ in xrange(self.params['n_silent_iterations'] - 1):
            stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
            self.MT.update_input(stim) 
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states[self.iteration_cnt, :] = state_
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            self.actions_taken[self.iteration_cnt, :] = next_action
            self.MT.advance_iteration()
            R = self.BG.get_reward_from_action(next_action[2], self.motion_params[self.iteration_cnt, :4], training=False)
            self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
            self.iteration_cnt += 1


    def test_after_training(self, stim_params):

        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., self.params['d1_gain_after_training'], self.params['param_msn_d1']['gain'])
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., self.params['d2_gain_after_training'], self.params['param_msn_d2']['gain'])

        self.VI.current_motion_params = deepcopy(stim_params)
        self.BG.stop_supervisor()
        self.BG.stop_efference_copy()
        for i_ in xrange(self.params['n_iterations_per_stim'] - self.params['n_silent_iterations']):
            stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, [0., 0.])
            self.MT.update_input(stim) 
            if params['debug_mpn']:
                print 'Saving spike trains...'
                utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
            self.motion_params[self.iteration_cnt, :4] = self.VI.current_motion_params # store the current motion parameters before they get updated
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states[self.iteration_cnt, :] = state_
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            self.actions_taken[self.iteration_cnt, :] = next_action
            self.MT.advance_iteration()
#            R = self.MT.get_reward_from_perceived_stim(state_)
            R = self.BG.get_reward_from_action(next_action[2], self.motion_params[self.iteration_cnt, :4], training=False)
            self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
            self.iteration_cnt += 1

        self.BG.stop_supervisor()
        self.BG.stop_efference_copy()
        # run 'silent iterations'
        for it_ in xrange(self.params['n_silent_iterations']):
            stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
            self.MT.update_input(stim) 
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states[self.iteration_cnt, :] = state_
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            self.actions_taken[self.iteration_cnt, :] = next_action
            self.MT.advance_iteration()
            R = self.BG.get_reward_from_action(next_action[2], self.motion_params[self.iteration_cnt, :4], training=False)
            self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
            self.iteration_cnt += 1



    def run_one_iteration(self, stim_params):
        # -------------------------------
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., self.params['params_synapse_d1_MT_BG']['gain'], self.params['param_msn_d1']['gain'])
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., self.params['params_synapse_d2_MT_BG']['gain'], self.params['param_msn_d2']['gain'])
        self.VI.current_motion_params = deepcopy(stim_params)
        self.BG.stop_supervisor()
        self.BG.stop_efference_copy()
        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, [0., 0.])
        self.MT.update_input(stim) 
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        self.motion_params[self.iteration_cnt, :4] = self.VI.current_motion_params # store the current motion parameters before they get updated
        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states[self.iteration_cnt, :] = state_
        next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        self.actions_taken[self.iteration_cnt, :] = next_action
        self.MT.advance_iteration()
        R = self.BG.get_reward_from_action(next_action[2], self.motion_params[self.iteration_cnt, :4], training=False)
        self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
        self.iteration_cnt += 1
        return R
        


    def run_test(self, stim_params):
        # -------------------------------
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., self.params['params_synapse_d1_MT_BG']['gain'], self.params['param_msn_d1']['gain'])
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., self.params['params_synapse_d2_MT_BG']['gain'], self.params['param_msn_d2']['gain'])
        self.VI.current_motion_params = deepcopy(stim_params)

        self.BG.stop_supervisor()
        self.BG.stop_efference_copy()
        for it_ in xrange(self.params['n_iterations_per_stim'] - self.params['n_silent_iterations']):
            stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, [0., 0.])
            self.MT.update_input(stim) 
            if params['debug_mpn']:
                print 'Saving spike trains...'
                utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
            self.motion_params[self.iteration_cnt, :4] = self.VI.current_motion_params # store the current motion parameters before they get updated
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states[self.iteration_cnt, :] = state_
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            self.actions_taken[self.iteration_cnt, :] = next_action
            self.MT.advance_iteration()
            R = self.BG.get_reward_from_action(next_action[2], self.motion_params[self.iteration_cnt, :4], training=False)
            self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
            self.iteration_cnt += 1
            
        # run 'silent iterations'
        for it_ in xrange(self.params['n_silent_iterations']):
            stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
            self.MT.update_input(stim) 
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states[self.iteration_cnt, :] = state_
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            self.actions_taken[self.iteration_cnt, :] = next_action
            self.MT.advance_iteration()
            R = self.BG.get_reward_from_action(next_action[2], self.motion_params[self.iteration_cnt, :4], training=False)
            self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
            self.iteration_cnt += 1


    def trigger_pre_spikes(self):
        # -------------------------------
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., 0., 0.)
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., 0., 0.)
        self.VI.current_motion_params = deepcopy(stim_params)

        self.BG.stop_supervisor()
        self.BG.stop_efference_copy()
        stim = self.VI.spikes_for_all(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        nest.Simulate(self.params['t_iteration'])
        self.iteration_cnt += 1


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
#        self.training_stimuli = np.zeros((params['n_stim'], 4))

#        for i_ in xrange(self.params['n_stim']):
#            self.training_stimuli[i_, :] = self.params['initial_state']
        self.VI.create_training_sequence_RBL(self.BG)
#        if pc_id == 0:
#            np.savetxt(params['training_sequence_fn'], self.training_stimuli)
        if self.comm != None:
            self.comm.Barrier()
        self.action_indices = []
        supervisor_states, action_indices, motion_params_precomputed = self.VI.get_supervisor_actions(self.VI.training_stimuli, self.BG)
#        print 'supervisor_states:', supervisor_states
#        print 'action_indices:', action_indices

        np.savetxt(params['supervisor_states_fn'], supervisor_states)
#        np.savetxt(params['action_indices_fn'], action_indices, fmt='%d')
        np.savetxt(params['motion_params_precomputed_fn'], motion_params_precomputed)

#        self.action_indices = np.zeros(self.params['n_stim'], dtype=np.int)
#        self.supervisor_states, self.action_indices, self.motion_params_precomputed = self.VI.get_supervisor_actions(self.training_stimuli, self.BG)
        self.rewards = np.zeros(params['n_iterations'] + 1) # + 1 because rewards are available after the iteration and will affect the next 
#        print 'self.training_stimuli:', self.training_stimuli
#        print 'self.training_stimuli.shape', self.training_stimuli.shape
#        print 'self.supervisor_states', self.supervisor_states
        # unnecessary as it will be overwritten
        self.motion_params[:, 4] = np.arange(0, self.params['n_iterations'] * self.params['t_iteration'], self.params['t_iteration'])


    def save_data_structures(self):
        if pc_id == 0:
            utils.remove_empty_files(self.params['connections_folder'])
            utils.remove_empty_files(self.params['spiketimes_folder'])
#            np.savetxt(self.params['supervisor_states_fn'], self.supervisor_states)
#            np.savetxt(self.params['motion_params_precomputed_fn'], self.motion_params_precomputed)
            np.savetxt(self.params['actions_taken_fn'], self.actions_taken)
            np.savetxt(self.params['network_states_fn'], self.network_states)
            np.savetxt(self.params['rewards_given_fn'], self.rewards)
            np.savetxt(params['motion_params_fn'], self.motion_params)
            np.savetxt(params['activity_memory_fn'], self.BG.activity_memory)
            np.savetxt(params['action_indices_fn'], np.array(self.action_indices))


    if comm != None:
        comm.Barrier()




if __name__ == '__main__':

    write_params = True
    GP = simulation_parameters.global_parameters()
    if len(sys.argv) < 3:
        params = GP.params
    else:
        testing_params_json = utils.load_params(os.path.abspath(sys.argv[2]))
        params = utils.convert_to_NEST_conform_dict(testing_params_json)
        write_params = False
    
    if pc_id == 0 and write_params:
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation

    if pc_id == 0 and not write_params: # do not delete anything, when training is to be continued
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
#    CC.connect_mt_to_bg_RBL(RBL.MT, RBL.BG, training_params, params, target='d1', model=params['synapse_d1_MT_BG'])
#    CC.connect_mt_to_bg_RBL(RBL.MT, RBL.BG, training_params, params, target='d2', model=params['synapse_d2_MT_BG'])
    RBL.CC.connect_mt_to_bg(RBL.MT, RBL.BG)

#    stim_type = []
#    d1_actions_trained = []
#    d2_actions_trained = []
#    all_actions_trained = []
#    speeds_trained = []

    i_stim = 0
    v_stim_cnt = 0

    times = []
    for i_cycle in xrange(RBL.params['n_training_cycles']):
        for i_v in xrange(RBL.params['n_training_v']):

            # take the new stimulus from the mixed distribution as derived above
            stim_params = RBL.VI.training_stimuli[v_stim_cnt, :]
#            stim_params[2] = RBL.VI.training_stimuli[v_stim_cnt, 2]
            print 'DEBUG i_stim %d v_stim_cnt %d stim_params:' % (i_stim, v_stim_cnt), stim_params

            for i_x in xrange(RBL.params['n_training_x']):
                for i_neg in xrange(RBL.params['suboptimal_training']):
                    # train D2
                    (required_v_eye, v_y, action_idx) = RBL.BG.get_non_optimal_action_for_stimulus(stim_params)
                    RBL.motion_params[i_stim, :4] = deepcopy(stim_params)
                    action_v = [required_v_eye, 0.]
                    t_ = time.time()
                    RBL.train_doing_action_with_supervisor(RBL.motion_params[i_stim, :4], action_v, v_eye=[0., 0.])
                    times.append(time.time() - t_)
                    i_stim += 1
                RBL.BG.reset_pool_of_possible_actions()
                # one training with the correct / optimal action, train D1
                (required_v_eye, v_y, action_idx) = RBL.BG.get_optimal_action_for_stimulus(stim_params)
                action_v = [required_v_eye, 0.]
                RBL.motion_params[i_stim, :4] = deepcopy(stim_params)
                t_ = time.time()
                RBL.train_doing_action_with_supervisor(RBL.motion_params[i_stim, :4], action_v, v_eye=[0., 0.])
                times.append(time.time() - t_)
                stim_params = utils.get_next_stim(RBL.params, stim_params, required_v_eye) # follow the stimulus to the center and update the stim params with the new ones
                stim_params = list(stim_params)
                i_stim += 1
            v_stim_cnt += 1

    
    f_time = file('%stimes_training_%d.json' % (params['tmp_folder'], pc_id), 'w')
    json.dump(times, f_time, indent=2)
    f_time.flush()
    f_time.close()
#    print 'DEBUG stim_type:', stim_type
#    print 'DEBUG d1_actions_trained', d1_actions_trained
#    print 'DEBUG d2_actions_trained', d2_actions_trained
#    print 'DEBUG all_actions_trained', all_actions_trained

    # in order to update the weights, switch off kappa and trigger pre-synaptic spikes in ALL cells
    # TESTING: one cycle
    print 'trigger_pre_spikes'
    RBL.trigger_pre_spikes()

#    for iter_stim in xrange(params['n_training_stim_per_cycle']):
#        stim_params = deepcopy(RBL.training_stimuli[iter_stim, :])
#        RBL.run_test(stim_params)

    RBL.save_data_structures()
    RBL.CC.get_weights(RBL.MT, RBL.BG)
    RBL.CC.get_d1_d1_weights(RBL.BG)
    RBL.CC.get_d2_d2_weights(RBL.BG)

    if pc_id == 0:
        if params['n_stim'] > 6:
            n_stim = 6 
        else:
            n_stim = params['n_stim']
        run_plot_bg(params, (0, n_stim))
        MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(n_stim)])
        MAC = MetaAnalysisClass([params['folder_name']])
        run_plot_bg(params, None)

    t1 = time.time()
    print 'Time pc_id %d: %d [sec] %.1f [min]' % (pc_id, t1 - t0, (t1 - t0)/60.)

