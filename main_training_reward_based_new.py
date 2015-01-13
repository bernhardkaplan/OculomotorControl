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

#        if self.params['mixed_training_cycles']:
#            self.training_stim_params = self.VI.create_training_sequence_RBL_mixed_within_a_cycle()
#        else:
#            self.training_stim_params = self.VI.create_training_sequence_RBL_cycle_blocks()
#        self.training_stimuli = self.VI.get_training_stimuli() # this is to be done 'externally' with a given file of training parameters
        if self.comm != None:
            self.comm.Barrier()


    def save_data_structures(self):
        if pc_id == 0:
            print 'DEBUG, removing empty files from:', self.params['connections_folder']
            print 'DEBUG, removing empty files from:', self.params['spiketimes_folder']
            utils.remove_empty_files(self.params['connections_folder'])
            utils.remove_empty_files(self.params['spiketimes_folder'])
            np.savetxt(self.params['actions_taken_fn'], np.array(self.actions_taken))
            np.savetxt(self.params['network_states_fn'], np.array(self.network_states))
            np.savetxt(self.params['rewards_given_fn'], np.array(self.rewards))
            np.savetxt(params['motion_params_training_fn'], np.array(self.motion_params))
            np.savetxt(params['K_values_fn'], np.array(self.K_vec))


    def prepare_training(self, old_params=None):
        self.create_networks()
        self.set_up_data_structures()
        if old_params == None:
            self.CC.connect_mt_to_bg(self.MT, self.BG)
        else:
            print 'Loading weight matrix for D1'
            self.CC.connect_and_load_mt_to_bg(self.MT, self.BG, 'd1', old_params)
            print 'Loading weight matrix for D2'
            self.CC.connect_and_load_mt_to_bg(self.MT, self.BG, 'd2', old_params)



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
        next_action = self.BG.get_action_softmax()
        self.network_states.append(state_)
        self.advance_iteration()
        self.K_vec.append(0)

        ##############################################
        #    A C T I O N    R E A D O U T 
        ##############################################
        # For evaluation of the action, shift the stimulus back in time
        x_pre_action = stim_params[0] - self.params['delay_input'] * stim_params[2] / 1000. 
        stim_params_evaluation = (x_pre_action, stim_params[1], stim_params[2], stim_params[3])
        new_stim = utils.get_next_stim(params, stim_params_evaluation, next_action[0])
#        R = utils.get_reward_from_perceived_states(x_old, new_stim[0])
#        R = self.BG.get_binary_reward(stim_params, next_action[2])
#        R = utils.get_reward_gauss(new_stim[0], stim_params)#, params)
        # this is the trick: the system knows about delay_input
        # If it was: R = utils.get_reward_sigmoid(new_stim[0], stim_params, params) delay_input would not be taken into account
        R = utils.get_reward_sigmoid(new_stim[0], stim_params_evaluation, params) 
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
        return next_action, R
        if self.comm != None:
            self.comm.Barrier()



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
    old_params = None
    GP = simulation_parameters.global_parameters()
    trained_stimuli = []
    info_txt = "\nThere are different use cases:\n \
    \tpython script_name [training_stimuli_fn] [training_stim_idx] \
    \tpython script_name [folder_containing_connectivity] [training_stimuli_fn] [training_stim_idx] \
    "
    assert len(sys.argv) > 2, 'Missing training_stim_information and training_stim_idx!' + info_txt
    if len(sys.argv) == 3:
        training_params_fn = sys.argv[1]
        continue_training_idx = int(sys.argv[2])
        params = GP.params
    elif len(sys.argv) == 4:
        print 'Loading old parameters file from:', sys.argv[1]
        old_params_json = utils.load_params(os.path.abspath(sys.argv[1]))
        old_params = utils.convert_to_NEST_conform_dict(old_params_json)
        training_params_fn = sys.argv[2]
        continue_training_idx = int(sys.argv[3])
        params = GP.params
        if comm != None:
            comm.Barrier()
    elif len(sys.argv) == 5:
        print 'Loading old parameters file from:', sys.argv[1]
        old_params_json = utils.load_params(os.path.abspath(sys.argv[1]))
        old_params = utils.convert_to_NEST_conform_dict(old_params_json)

        if comm != None:
            comm.Barrier()
        print 'Loading current parameter file from:', sys.argv[2]
        params_json = utils.load_params(os.path.abspath(sys.argv[2]))
        params = utils.convert_to_NEST_conform_dict(params_json)
        # load already trained stimuli
        trained_stimuli = old_params['trained_stimuli']
        training_params_fn = sys.argv[3]
        continue_training_idx = int(sys.argv[4])
    else:
        print 'Wrong number of sys.argv!', info_txt
        exit(1)

    if pc_id == 0:
        print 'DEBUG sys.argv', sys.argv, 'continue_training_idx', continue_training_idx

    training_params = np.loadtxt(training_params_fn)
    n_max = continue_training_idx + params['n_training_cycles'] * params['n_training_stim_per_cycle']
    assert (training_params[:, 0].size >= n_max), 'The expected number of training iterations (= %d) is too high for the given training_params from file %s (contains %d training stim)' % \
            (n_max, training_params_fn, training_params[:, 0].size)

    params['training_params_fn'] = training_params_fn
    if pc_id == 0:
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
    if pc_id == 0:
        print 'DEBUG: removing files from:', params['spiketimes_folder']
        print 'DEBUG: removing files from:', params['input_folder_mpn']
        print 'DEBUG: removing files from:', params['connections_folder']
        utils.remove_files_from_folder(params['spiketimes_folder'])
        utils.remove_files_from_folder(params['input_folder_mpn'])
        utils.remove_files_from_folder(params['connections_folder'])
        np.savetxt(params['training_stimuli_fn'], training_params)
        print 'DEBUG training params:', training_params

    if comm != None:
        comm.Barrier()
    t0 = time.time()

    ###################
    #    S E T   U P 
    ###################
    RBL = RewardBasedLearning(params, comm)
    RBL.prepare_training(old_params)
    #if old_params != None and params['continue_training']:
    RBL.training_stimuli = training_params

    # keep track of trained stimuli and d1/d2 actions that have been trained
    # python 2.6
    d1_actions_trained = {}
    d2_actions_trained = {}
    for i in xrange(n_max):
        d1_actions_trained[i] = []
        d2_actions_trained[i] = []
    # python 2.7
    #d1_actions_trained = { i : [] for i in xrange(params['n_stim'])}
    #d2_actions_trained = { i : [] for i in xrange(params['n_stim'])}

    unsuccessfully_trained_stimuli = []
    n_training_trials = 0 
    ####################################
    #   T R A I N   A   S T I M U L U S 
    ####################################
    #TODO:
    for i_cycle in xrange(params['n_training_cycles']):
        print '\n================ NEW CYCLE ======================'
        # randomize order of stimuli within each cycle
        order_of_stim = range(params['n_training_stim_per_cycle'])
        if i_cycle > 0:
            np.random.shuffle(order_of_stim) 

        #actions_per_stim = [{a: 0 for a in xrange(params['n_actions'])} for i in xrange(params['n_training_stim_per_cycle'])] 
        actions_per_stim = []
        #for i in xrange(params['n_training_stim_per_cycle']):
        for i in xrange(n_max):
            d = {}
            for a in xrange(params['n_actions']):
                d[a] = 0
            actions_per_stim.append(d)

        for i_ in xrange(params['n_training_stim_per_cycle']):
            # pick a stimulus to train
            if old_params == None or not params['continue_training']:
                i_stim = order_of_stim[i_]
            else:
                i_stim = continue_training_idx + i_cycle * params['n_training_stim_per_cycle'] + i_
            stim_params = RBL.training_stimuli[i_stim, :]
            print 'stim_params for i_stim %d' % i_stim, stim_params
            # reinitialize the counters how often an action has been selected for each stimulus
            cnt_trial = 0  # counts the total number of trials for any action (including pos and neg reward trials)
            trained_stimuli.append((i_stim, list(stim_params)))
            while (cnt_trial < params['n_max_trials_same_stim']): # independent of rewards

                v_and_action, R = RBL.present_stimulus_and_train(stim_params)
                n_training_trials += 1
                trained_action = v_and_action[2]
                actions_per_stim[i_stim][trained_action] += 1

                cnt_trial += 1
                if cnt_trial >= params['n_max_trials_same_stim']:
                    unsuccessfully_trained_stimuli.append(i_stim)
                if (actions_per_stim[i_stim][trained_action] >= params['n_max_trials_pos_rew'] and R > 0): 
                    d1_actions_trained[i_stim].append(trained_action)
                    # new stimulus!
                    i_stim += 1
                    cnt_trial = 0
                    print 'Ending training for this stimulus'
                    break
                elif (R < 0):
                    d2_actions_trained[i_stim].append(trained_action)

    if len(unsuccessfully_trained_stimuli) > 0:
        np.savetxt(params['data_folder'] + 'unsuccessfully_trained_stimuli.dat', np.array(unsuccessfully_trained_stimuli))

    # update the trained_stimuli parameter in the Parameters/simulation_parameters.json file
    params['n_training_trials'] = n_training_trials
    params['trained_stimuli'] = trained_stimuli
    params['d1_actions_trained'] = d1_actions_trained
    params['d2_actions_trained'] = d2_actions_trained
    params['training_stim_offset'] = continue_training_idx
    if comm != None:
        comm.Barrier()
    if pc_id == 0:
        GP.write_parameters_to_file(params['params_fn_json'], params)
    if comm != None:
        comm.Barrier()

    ######################################
    #    TRIGGER SPIKES
    #######################################
    RBL.trigger_pre_spikes()

    ####################################
    #   S A V E     W E I G H T S 
    ####################################
    t_a = time.time()
    RBL.CC.get_weights(RBL.MT, RBL.BG)
    RBL.CC.get_d1_d1_weights(RBL.BG)
    RBL.CC.get_d2_d2_weights(RBL.BG)
    RBL.CC.merge_connection_files(params)
    t_b = time.time() - t_a
    print 'Time for get_weights %d: %.2f [sec] %.2f [min]' % (pc_id, t_b, t_b / 60.)

    ####################################
    #   R U N   E M P T Y    I N P U T 
    ####################################
#    RBL.run_without_stimulus()

    ################################
    #   T E S T    S T I M U L U S 
    ################################
#    RBL.test_after_training(RBL.training_stimuli[0, :])
#    RBL.test_after_training(stim_params) 

    RBL.save_data_structures()

    t1 = time.time() - t0
    print 'n_iterations: RBL', RBL.iteration_cnt
    print 'n_iterations: MPN', RBL.MT.iteration
    print 'n_iterations: BG', RBL.BG.iteration
    print 'n_iterations: VI', RBL.VI.iteration
    print 'TimeEND %d: %.2f [sec] %.2f [min]' % (pc_id, t1, t1 / 60.)
    if comm != None:
        comm.Barrier()

    #####################
    #   P L O T T I N G 
    #####################
    if not params['Cluster']:
        from PlottingScripts.PlotBGActivity import run_plot_bg
        from PlottingScripts.PlotMPNActivity import MetaAnalysisClass
        from PlottingScripts.SuperPlot import PlotEverything
        if pc_id == 0:
            n_stim = 1
            print 'Running analysis...'
            P = PlotEverything(['dummy_cmd', params['folder_name']], verbose=True)
            run_plot_bg(params, None)
    #        run_plot_bg(params, (0, n_stim))
            MAC = MetaAnalysisClass([params['folder_name']])
            MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(n_stim)])

