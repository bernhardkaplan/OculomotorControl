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
from PlottingScripts.SuperPlot import PlotEverything

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
    
        self.network_states = []        # MPN readout
        self.actions_taken = []         # BG action output
        self.rewards = []               # rewards given
        self.motion_params = []         # motion parameters for stimulus presentation
        self.K_vec = []

        if self.params['mixed_training_cycles']:
            self.training_stim_params = self.VI.create_training_sequence_RBL_mixed_within_a_cycle()
        else:
            self.training_stim_params = self.VI.create_training_sequence_RBL_cycle_blocks()
        if self.comm != None:
            self.comm.Barrier()


    def save_data_structures(self):
        if pc_id == 0:
            utils.remove_empty_files(self.params['connections_folder'])
            utils.remove_empty_files(self.params['spiketimes_folder'])
            np.savetxt(self.params['actions_taken_fn'], np.array(self.actions_taken))
            np.savetxt(self.params['network_states_fn'], np.array(self.network_states))
            np.savetxt(self.params['rewards_given_fn'], np.array(self.rewards))
            np.savetxt(params['motion_params_training_fn'], np.array(self.motion_params))
            np.savetxt(params['K_values_fn'], np.array(self.K_vec))


    def prepare_training(self, w_init_fn=None): 
        self.create_networks()
        self.set_up_data_structures()
        if w_init_fn == None:
            # TODO: load a weight matrix in order to continue training
            self.CC.connect_mt_to_bg(self.MT, self.BG)
        else:
            self.CC.connect_and_load_mt_to_bg(self.MT, self.BG, w_init)


    def create_networks(self):
        self.VI = VisualInput.VisualInput(self.params, comm=self.comm)
        self.MT = MotionPrediction.MotionPrediction(self.params, self.VI, self.comm)
        self.BG = BasalGanglia.BasalGanglia(self.params, self.comm)
        self.CC = CreateConnections.CreateConnections(self.params, self.comm)


    def present_stimulus_and_train(self, stim_params):
        #######################################
        # 1   N O I S E    R U N 
        #######################################
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., self.params['gain_MT_d1'], 0.)
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., self.params['gain_MT_d2'], 0.)
        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        nest.Simulate(self.params['t_iteration'])
        self.advance_iteration()
        self.K_vec.append(0)

        ######################################
        # 2   S T I M    P R E S E N T A T I O N 
        #######################################
        self.VI.current_motion_params = deepcopy(stim_params)
        self.motion_params.append(deepcopy(stim_params))
        stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, [0., 0.]) # assume a still eye with speed = [0., 0.]
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        nest.Simulate(params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states.append(state_)
#        next_action = self.BG.get_action(WTA=True) # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        next_action = self.BG.get_action(WTA=False) # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
        self.network_states.append(state_)
        self.advance_iteration()
        self.K_vec.append(0)

        ##############################################
        #    A C T I O N    R E A D O U T 
        ##############################################
        x_old = stim_params[0]
        new_stim = utils.get_next_stim(params, stim_params, next_action[0])
#        R = utils.get_reward_from_perceived_states(x_old, new_stim[0])
        R = self.BG.get_binary_reward(stim_params, next_action[2])
        self.rewards.append(R)
        self.actions_taken.append([next_action[0], next_action[1], next_action[2], R])
#        print 'Reward:', R
        self.BG.activate_efference_copy(np.int(np.round(next_action[2])))

        #######################################
        # 3   S I L E N T / N O I S E    R U N: wait for the consequence of the previous action (next stimulus is not gated to perception)
        #######################################
        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        nest.Simulate(self.params['t_iteration'])
        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        self.network_states.append(state_)
        self.advance_iteration()
        self.K_vec.append(0)

        #######################
        #     L E A R N I N G 
        # 4 + n_iterations_RBL_training 
        #######################
        if R >= 0:
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, R, 0., 0.) # kappa, syn_gain, bias_gain
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., 0., 0.) 
        else:
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., 0., 0.)
            self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, -R, 0., 0.)
        nest.Simulate(self.params['n_iterations_RBL_training'] * params['t_iteration']) 
        for i_ in xrange(self.params['n_iterations_RBL_training']):
            self.advance_iteration()
            self.K_vec.append(R)

        #######################################
        #    S I L E N T / N O I S E    R U N 
        # 5 + n_iterations_RBL_training   
        #######################################
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., self.params['gain_MT_d1'], 0.)
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., self.params['gain_MT_d2'], 0.)
        self.BG.stop_efference_copy()
        self.BG.stop_supervisor()
        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        nest.Simulate(self.params['t_iteration'])
#        state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
#        self.network_states.append(state_)
        self.advance_iteration()
        self.K_vec.append(0)
        return next_action



    def run_without_stimulus(self):
        ######################################
        #    EMPTY RUNS
        #######################################
        self.BG.stop_efference_copy()
        self.BG.stop_supervisor()
        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        nest.Simulate(self.params['t_iteration'])
        self.advance_iteration()



    def test_after_training(self, stim_params):

        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., self.params['d1_gain_after_training'], self.params['param_msn_d1']['gain'])
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., self.params['d2_gain_after_training'], self.params['param_msn_d2']['gain'])

        self.VI.current_motion_params = deepcopy(stim_params)
        self.BG.stop_supervisor()
        self.BG.stop_efference_copy()
        for i_ in xrange(self.params['n_iterations_per_stim'] - 1):
            stim, supervisor_state = self.VI.compute_input(self.MT.local_idx_exc, [0., 0.])
            self.MT.update_input(stim) 
            if params['debug_mpn']:
                print 'Saving spike trains...'
                utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
            print 'DEBUG, pc_id %d time %.2f stim' % (self.pc_id, nest.GetKernelStatus()['time']), stim
            self.motion_params.append(self.VI.current_motion_params)
            nest.Simulate(self.params['t_iteration'])
            state_ = self.MT.get_current_state(self.VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
            self.network_states.append(state_)
            print 'debug iteration_cnt:', self.iteration_cnt, 'bg.iteration', self.BG.iteration, 'params[n_iterations]:', self.params['n_iterations']
            next_action = self.BG.get_action() # read out the activity of the action population, necessary to fill the activity memory --> used for efference copy
            R = utils.get_reward_from_perceived_states(stim_params[0], state_[0])
            self.rewards.append(R)
            self.actions_taken.append([next_action[0], next_action[1], next_action[2], R])
            self.advance_iteration()

        # run 'silent iterations'
        stim, supervisor_state = self.VI.set_empty_input(self.MT.local_idx_exc)
        self.MT.update_input(stim) 
        nest.Simulate(self.params['t_iteration'])
        self.advance_iteration()



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
        



    def trigger_pre_spikes(self):
        # -------------------------------
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD1, 0., 0., 0.)
        self.BG.set_kappa_and_gain(self.MT.exc_pop, self.BG.strD2, 0., 0., 0.)

        self.BG.stop_supervisor()
        self.BG.stop_efference_copy()
        stim = self.VI.spikes_for_all(self.MT.local_idx_exc)
#        self.MT.update_input(stim) 
        self.MT.update_trigger_spikes(stim)
        if params['debug_mpn']:
            print 'Saving spike trains...'
            utils.save_spike_trains(self.params, self.iteration_cnt, stim, self.MT.local_idx_exc)
        nest.Simulate(self.params['t_iteration'])
        self.advance_iteration()
        self.K_vec.append(0)


    def advance_iteration(self):
        self.MT.advance_iteration()
        self.BG.advance_iteration()
        self.VI.advance_iteration()
        self.iteration_cnt += 1




if __name__ == '__main__':

    t0 = time.time()
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
    #TODO:
    i_stim  = 0
    for i_cycle in xrange(params['n_training_cycles']):
        for i_v in xrange(params['n_training_v']):
            for i_trials_per_speed in xrange(params['n_training_x']):
#                stim_params = RBL.training_stim_params[i_stim, :]
                stim_params = RBL.training_stim_params[0, :]
                print 'stim_params for i_stim %d' % i_stim, stim_params
                trained_action = RBL.present_stimulus_and_train(stim_params)
                i_stim += 1


    ######################################
    #    TRIGGER SPIKES
    #######################################
    RBL.trigger_pre_spikes()

    ####################################
    #   S A V E     W E I G H T S 
    ####################################
    RBL.CC.get_weights(RBL.MT, RBL.BG)

    ####################################
    #   R U N   E M P T Y    I N P U T 
    ####################################
#    RBL.run_without_stimulus()

    ################################
    #   T E S T    S T I M U L U S 
    ################################
#    RBL.test_after_training(RBL.training_stim_params[0, :])
#    RBL.test_after_training(stim_params) 

    RBL.save_data_structures()

    t1 = time.time() - t0
    print 'TimeEND: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    print 'n_iterations: RBL', RBL.iteration_cnt
    print 'n_iterations: MPN', RBL.MT.iteration
    print 'n_iterations: BG', RBL.BG.iteration
    print 'n_iterations: VI', RBL.VI.iteration

    #####################
    #   P L O T T I N G 
    #####################
    if pc_id == 0:
        n_stim = 1
        print 'Running analysis...'
        P = PlotEverything(sys.argv, verbose=True)
        run_plot_bg(params, None)
#        run_plot_bg(params, (0, n_stim))
        MAC = MetaAnalysisClass([params['folder_name']])
        MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(n_stim)])

