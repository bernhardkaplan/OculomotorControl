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
        self.retrained_actions = []
        
    def prepare_training(self, w_init_fn=None): 
        self.create_networks()
        self.set_up_data_structures()
        if w_init_fn != None:
            # TODO: load a weight matrix in order to continue training
            self.CC.
            
    RBL.CC.connect_mt_to_bg(RBL.MT, RBL.BG)

    def create_networks(self):
        self.VI = VisualInput.VisualInput(self.params, comm=self.comm)
        self.MT = MotionPrediction.MotionPrediction(self.params, self.VI, self.comm)
        self.BG = BasalGanglia.BasalGanglia(self.params, self.comm)
        self.CC = CreateConnections.CreateConnections(self.params, self.comm)


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
        self.CC.get_weights(self.MT, self.BG, iteration=self.iteration_cnt)

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
    if pc_id == 0:
        utils.remove_files_from_folder(params['spiketimes_folder'])
        utils.remove_files_from_folder(params['input_folder_mpn'])
        utils.remove_files_from_folder(params['connections_folder'])
    if comm != None:
        comm.Barrier()
    t0 = time.time()


    ###################
    #    S E T   U P 
    ###################
    RBL = RewardBasedLearning(params, comm)
    RBL.prepare_training()


    ####################################
    #   T R A I N   A   S T I M U L U S 
    ####################################
    RBL.present_stimulus_and_train()

    ######################################
    #    N O I S E    R U N S
    #######################################
    for i_ in xrange(4):
        nest.Simulate(params['t_iteration']) 
        stim, supervisor_state = RBL.VI.set_empty_input(RBL.MT.local_idx_exc)
        RBL.iteration_cnt += 1
        next_action = RBL.BG.get_action(WTA=True)
        v_eye = [next_action[0], next_action[1]]
        print 'next action:', next_action


    ######################################
    #
    #    S T I M    P R E S E N T A T I O N 
    #
    #######################################
    stim_params = list(params['initial_state'])
    RBL.VI.current_motion_params = stim_params
    stim, supervisor_state = RBL.VI.compute_input(RBL.MT.local_idx_exc, v_eye)
    utils.save_spike_trains(RBL.params, RBL.iteration_cnt, stim, RBL.MT.local_idx_exc)
    RBL.MT.update_input(stim) 
    RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD1, 0., 1., 0.)
    RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD2, 0., 1., 0.)
    nest.Simulate(params['t_iteration'])
    RBL.iteration_cnt += 1


    ######################################
    #    ACTION READOUT 
    #######################################
    x_old = stim_params[0]
    stim_params = utils.get_next_stim(params, stim_params, v_eye[0])
    R = utils.get_reward_from_perceived_states(x_old, stim_params[0])
    print 'Reward:', R


    ######################################
    #    LEARNING 
    #######################################
    if R >= 0:
        RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD1, R, 0., 0.) # kappa, syn_gain, bias_gain
        RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD2, 0., 0., 0.) 
    else:
        RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD1, 0., 0., 0.)
        RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD2, -R, 0., 0.)
    RBL.BG.activate_efference_copy(next_action[2])
    nest.Simulate(1 * params['t_iteration']) 
    RBL.iteration_cnt += 1


    ######################################
    #    LEARNING 
    #######################################
    print 'trigger_pre_spikes'
    RBL.trigger_pre_spikes()


    ######################################
    #    EMPTY RUNS
    #######################################
    RBL.BG.stop_efference_copy()
    RBL.BG.stop_supervisor()
    stim, supervisor_state = RBL.VI.set_empty_input(RBL.MT.local_idx_exc)
    RBL.MT.update_input(stim) 
    nest.Simulate(RBL.params['t_iteration'])
    RBL.iteration_cnt += 1

    # TESTING
    RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD1, 0., 1., 0.)
    RBL.BG.set_kappa_and_gain(RBL.MT.exc_pop, RBL.BG.strD2, 0., 1., 0.)

    v_eye = [0., 0.]
    stim_params = list(params['initial_state'])
    RBL.VI.current_motion_params = stim_params
    stim, supervisor_state = RBL.VI.compute_input(RBL.MT.local_idx_exc, v_eye)
    utils.save_spike_trains(RBL.params, RBL.iteration_cnt, stim, RBL.MT.local_idx_exc)
    print 'debug stim:', stim
    RBL.MT.update_input(stim) 
    nest.Simulate(params['t_iteration'])
    RBL.iteration_cnt += 1

    next_action = RBL.BG.get_action()
    v_eye = [next_action[0], next_action[1]]
    print 'next action:', next_action
       
    print 'Iteration Count:', RBL.iteration_cnt

    RBL.CC.get_weights(RBL.MT, RBL.BG, iteration=RBL.iteration_cnt)
