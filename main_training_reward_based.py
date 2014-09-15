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
        
    def set_connection_module(self, CM):
        self.CM = CM

    def create_networks(self):
        self.VI = VisualInput.VisualInput(self.params, comm=self.comm)
        self.MT = MotionPrediction.MotionPrediction(self.params, self.VI, self.comm)
        self.VI.set_pc_id(pc_id)
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


    def run_doing_action(self, stim_params, action, K=0, gain=1., v_eye=[0., 0.]):
        """
        Simulate two iterations with one pre-determined action in between:
            1) Stimulus with stim_params is perceived and action is done as reaction to that
            2) A new stimulus with modified stimulus parameters is perceived and simulated
        Arguments:
            stim_params     --  (x, y, v_x, v_y)
            action          --  (v_x, v_y)
            K               --  Kappa
            gain            --  float
            v_eye           --  the initial speed of the eye (influences the perceived stimulus speed in the first iteration)
        """

#        print 'DEBUG, iteration %d VI.t_current ' % self.iteration_cnt, self.VI.t_current, 'current_motion_params', self.VI.current_motion_params
        R = 0.
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
        (action_index_x, action_index_y) = self.BG.supervised_training(action)

        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        self.MT.update_input(stim) # run the network for some time 

        # for the first two iterations switch off plasticity and set gain to 1
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, kappa=0., syn_gain=self.params['params_synapse_d1_MT_BG']['gain'], bias_gain=self.params['param_msn_d1']['gain'])
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, kappa=0., syn_gain=self.params['params_synapse_d2_MT_BG']['gain'], bias_gain=self.params['param_msn_d2']['gain'])

        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states[self.iteration_cnt, :] = state_

        next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        self.retrained_actions.append(next_action)
        v_eye[0] = next_action[0]
        v_eye[1] = next_action[1]
        self.actions_taken[self.iteration_cnt, :] = next_action
        self.MT.advance_iteration()
        self.iteration_cnt += 1

        self.CM.get_weights(self.MT, self.BG, iteration=self.iteration_cnt)


        #####
        # 2 # Do the action that has been decided in the previous iteration (or trigger a 'random' action)
        #####
        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, v_eye)
        self.MT.update_input(stim) # run the network for some time 
        self.BG.stop_supervisor()
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        self.motion_params[self.iteration_cnt, :4] = self.VI.current_motion_params # store the current motion parameters before they get updated

        # activate the efference copy in order to make correlation-binding between visual neurons and D2 / D1 neurons more likely
        self.BG.activate_efference_copy(self.iteration_cnt - 1, self.iteration_cnt)

        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states[self.iteration_cnt, :] = state_
        next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        v_eye[0] = next_action[0]
        v_eye[1] = next_action[1]
        self.actions_taken[self.iteration_cnt, :] = next_action
        self.MT.advance_iteration()
        R = self.MT.get_reward_from_perceived_stim(state_)
        self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
        self.iteration_cnt += 1


        #####
        # 3 #
        #####
        # after the given action has been taken, run an empty iteration without any BG or MT activity to 'clean' the activity and avoid overlap between iterations
        self.BG.stop_supervisor()
        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        # Also switch on Kappa to let the traces evolve 
#        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, kappa=0., syn_gain=self.params['params_synapse_d1_MT_BG']['gain'], bias_gain=self.params['param_msn_d1']['gain'])
#        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, kappa=0., syn_gain=self.params['params_synapse_d2_MT_BG']['gain'], bias_gain=self.params['param_msn_d2']['gain'])


        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states[self.iteration_cnt, :] = state_
        next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        v_eye[0] = next_action[0]
        v_eye[1] = next_action[1]
        self.actions_taken[self.iteration_cnt, :] = next_action
        # don't update R, because the plasticity should act based on the initial movement
        self.rewards[self.iteration_cnt + 1] = R # + 1 because the reward will affect the next iteration
        self.MT.advance_iteration()
        self.iteration_cnt += 1
        

        #####
        # 4 #   L E A R N I N G    P H A S E 
        #####
        # Now, switch on kappa, set gain to zero
        # However, keep in mind, that the previous action was a mixture of
        #  - the desired action AND
        #  - the activity driven by the (probably non-optimal) MT-BG connectivity
        if R >= 0:
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, kappa=R, syn_gain=0, bias_gain=0.)
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, kappa=0., syn_gain=0, bias_gain=0.) 
        else:
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, kappa=0., syn_gain=0, bias_gain=0.)
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, kappa=-R, syn_gain=0, bias_gain=0.)
        self.BG.stop_supervisor()

        for i_ in xrange(self.params['n_iterations_per_stim'] - 3): # - 3 because of the 2 iterations for performing the action + 1 silent iteration
            stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
            self.MT.update_input(stim) 
            if i_ == (self.params['n_iterations_per_stim'] - 4):
                self.BG.stop_efference_copy()
                self.BG.stop_supervisor()
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states[self.iteration_cnt, :] = state_
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            v_eye[0] = next_action[0]
            v_eye[1] = next_action[1]
            self.actions_taken[self.iteration_cnt, :] = next_action
            # don't update R, because the plasticity should act based on the initial movement
            self.rewards[self.iteration_cnt] = R
            self.MT.advance_iteration()
            self.iteration_cnt += 1

        self.BG.stop_efference_copy()
        self.BG.stop_supervisor()


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

        for i_ in xrange(self.params['n_stim']):
            self.training_stimuli[i_, :] = self.params['initial_state']
#        self.training_stimuli = RBL.VI.create_training_sequence_RBL()

        self.action_indices = np.zeros(self.params['n_stim'], dtype=np.int)
#        self.supervisor_states, self.action_indices, self.motion_params_precomputed = self.VI.get_supervisor_actions(self.training_stimuli, self.BG)
        self.rewards = np.zeros(params['n_iterations'])
        print 'self.training_stimuli:', self.training_stimuli
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
            np.savetxt(self.params['action_indices_fn'], self.action_indices, fmt='%d')
            np.savetxt(self.params['actions_taken_fn'], self.actions_taken)
            np.savetxt(self.params['network_states_fn'], self.network_states)
            np.savetxt(self.params['rewards_given_fn'], self.rewards)
            np.savetxt(params['motion_params_fn'], self.motion_params)
            np.savetxt(params['training_sequence_fn'], self.training_stimuli)

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
    
    assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
    training_folder = os.path.abspath(sys.argv[1]) 
    print 'Training folder:', training_folder
    training_params_json = utils.load_params(training_folder)
    training_params = utils.convert_to_NEST_conform_dict(training_params_json)
    params['training_folder'] = training_folder
    if pc_id == 0 and write_params:
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
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
    CC = CreateConnections.CreateConnections(params, comm)
    RBL.set_connection_module(CC)
    CC.set_pc_id(pc_id)
    CC.connect_mt_to_bg_RBL(RBL.MT, RBL.BG, training_params, params, target='d1', model='bcpnn_synapse')
    CC.connect_mt_to_bg_RBL(RBL.MT, RBL.BG, training_params, params, target='d2', model='bcpnn_synapse')

#    if params['connect_d1_after_training']:
#        CC.connect_d1_after_training(BG, training_params, params)
#    RBL.CC.connect_mt_to_bg(RBL.MT, RBL.BG)

    # set the sequence of actions to be taken
#    action_sequence = np.zeros((params['n_stim'], 2))
#    action_sequence[0, :] = [10., 0.]
#    action_sequence[1, :] = [-3., 0.]
#    for i_stim in xrange(params['n_stim']):
#        stim_params = RBL.training_stimuli[i_stim, :]
#        RBL.run_doing_action(stim_params, action_sequence[i_stim, :])

    # simulate, do the given action and simulate again with the stimulus modified by the action
    action_v = [+10., 0]

    CC.get_weights(RBL.MT, RBL.BG, iteration=RBL.iteration_cnt)

    for iter_stim in xrange(params['n_stim']):
        # use the initial stimulus parameters 
        stim_params = deepcopy(RBL.training_stimuli[iter_stim, :])
#        print 'DEBUG RBL.training_stimuli before doing action', RBL.training_stimuli[iter_stim, :]
#        print 'DEBUG stim_params before doing action', stim_params
        RBL.run_doing_action(stim_params, action_v, v_eye=[0., 0.])
#        print 'DEBUG RBL.training_stimuli after doing action', RBL.training_stimuli[iter_stim, :]
#        print 'DEBUG stim_params after doing action', stim_params
#        CC.get_weights(RBL.MT, RBL.BG, iteration=RBL.iteration_cnt)


    print 'DEBUG stim_params after everything', RBL.training_stimuli
    RBL.save_data_structures()
    CC.get_weights(RBL.MT, RBL.BG)

    if pc_id == 0:
        run_plot_bg(params, (0, params['n_stim']))
        MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(params['n_stim'])])
        MAC = MetaAnalysisClass([params['folder_name']])
        run_plot_bg(params, None)

    t1 = time.time()
    print 'Time pc_id %d: %d [sec] %.1f [min]' % (pc_id, t1 - t0, (t1 - t0)/60.)
