import numpy as np
import nest
import utils
import json

class BasalGanglia(object):

    def __init__(self, params, comm=None, dummy=False):

        self.params = params
        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.comm = comm # mpi communicator needed to broadcast nspikes between processes
        if comm != None:
            assert (comm.rank == self.pc_id), 'mpi4py and NEST tell me different PIDs!'
            assert (comm.size == self.n_proc), 'mpi4py and NEST tell me different PIDs!'

        self.activity_memory = np.zeros((self.params['n_iterations'], self.params['n_actions']))
        self.RNG = np.random
        self.RNG.seed(self.params['basal_ganglia_seed'])
        self.create_suboptimal_action_mapping()
        self.iteration = 0
        self.set_action_speed_mapping_bins() 
        self.strD1 = {}
        self.strD2 = {}
        self.actions = {}
        self.recorder_output= {} # the actual NEST recorder object, indexed by naction
        self.gid_to_action = {} # here the key is the GID of the spike-recorder and the key is the action --> allows mapping of spike-GID --> action
        self.gid_to_action_D1 = {} # here the key is the GID of the spike-recorder and the key is the action --> allows mapping of spike-GID --> action
        self.gid_to_action_D2 = {} # here the key is the GID of the spike-recorder and the key is the action --> allows mapping of spike-GID --> action
        self.gid_to_action_via_spikerecorder= {} # here the key is the GID of the spike-recorder and the key is the action --> allows mapping of spike-GID --> action
        self.efference_copy = {}
        self.efference_copy_d1 = {}
        self.efference_copy_d2 = {}
        self.supervisor = {}
        # Recording devices
        self.recorder_d1 = {}
        self.recorder_d2 = {}
        self.recorder_states = {}
        self.recorder_efference = {}
        self.recorder_supervisor = {}

        self.voltmeter_d1 = {}
        self.voltmeter_d2 = {}
        self.voltmeter_rp = {}
        self.voltmeter_action = {}

        self.t_current = 0 
        self.currently_trained_action = None    # used for reward-based relearning

        self.bg_offset = {}
        self.bg_offset['d1'] = np.infty
        self.bg_offset['d2'] = np.infty
        self.bg_offset['actions'] = np.infty

        self.reset_pool_of_possible_actions()
        if not dummy:
            self.create_populations()
            if self.params['gids_to_record_bg']:
                self.record_extra_cells()

            if self.params['training']:
                self.connect_d1_population()
                if self.params['connect_d2_d2'] and self.params['with_d2']:
                    self.connect_d2_population()

            if self.params['connect_noise_to_bg']:
                self.connect_noise()


    def reset_pool_of_possible_actions(self):
        # for non optimal action selection
        self.all_action_idx = range(self.params['n_actions'])


    def record_extra_cells(self):

        self.params['gids_to_record_bg']
        self.voltmeter_extra = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :self.params['dt_volt']})
        nest.SetStatus(self.voltmeter_extra,[{"to_file": True, "withtime": True, 'label' : self.params['bg_volt_fn']}])
        nest.ConvergentConnect(self.voltmeter_extra, self.params['gids_to_record_bg']) # gids_to_record)


    def create_populations(self):
        #Creates D1 and D2 populations in STRIATUM, connections are created later
        for nactions in range(self.params['n_actions']):
            self.strD1[nactions] = nest.Create(self.params['model_exc_neuron'], self.params['num_msn_d1'], params=self.params['param_msn_d1'])
            for gid in self.strD1[nactions]:
                self.gid_to_action_D1[gid] = nactions
                self.bg_offset['d1'] = min(gid, self.bg_offset['d1'])

        if self.params['with_d2']:
            for nactions in range(self.params['n_actions']):
                self.strD2[nactions] = nest.Create(self.params['model_inh_neuron'], self.params['num_msn_d2'], params=self.params['param_msn_d2'])
                for gid in self.strD2[nactions]:
                    self.gid_to_action_D2[gid] = nactions
                    self.bg_offset['d2'] = min(gid, self.bg_offset['d2'])
        for nactions in range(self.params['n_actions']):
            if self.params['record_bg_volt']:
                self.voltmeter_d1[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :self.params['dt_volt']})
                nest.SetStatus(self.voltmeter_d1[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d1_volt_fn']+ str(nactions)}])
                if self.params['with_d2']:
                    self.voltmeter_d2[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :self.params['dt_volt']})
                    nest.SetStatus(self.voltmeter_d2[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d2_volt_fn']+ str(nactions)}])

        # Creates the different Populations, STR_D1, STR_D2 and Actions, and then create the Connections
        for nactions in range(self.params['n_actions']):
            self.actions[nactions] = nest.Create(self.params['model_bg_output_neuron'], self.params['num_actions_output'], params= self.params['param_bg_output'])
            for gid in self.actions[nactions]:
                self.gid_to_action[gid] = nactions

        for nactions in range(self.params['n_actions']):
            self.voltmeter_action[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :self.params['dt_volt']})
            if self.params['record_bg_volt']:
                nest.SetStatus(self.voltmeter_action[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['actions_volt_fn']+ str(nactions)}])
            self.recorder_output[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_action'])
            self.recorder_d1[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_d1'])
            if self.params['with_d2']: 
                self.recorder_d2[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_d2'])
            for ind in xrange(self.params['num_actions_output']):
                self.gid_to_action_via_spikerecorder[self.actions[nactions][ind]] = nactions
                self.bg_offset['actions'] = min(gid, self.bg_offset['actions'])
            nest.SetStatus(self.recorder_output[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['actions_spikes_fn']+ str(nactions)}])
            nest.SetStatus(self.recorder_d1[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d1_spikes_fn']+ str(nactions)}])
            if self.params['with_d2']: 
                nest.SetStatus(self.recorder_d2[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d2_spikes_fn']+ str(nactions)}])
            nest.ConvergentConnect(self.actions[nactions], self.recorder_output[nactions])
            nest.ConvergentConnect(self.strD1[nactions], self.recorder_d1[nactions])
            if self.params['with_d2']: 
                nest.ConvergentConnect(self.strD2[nactions], self.recorder_d2[nactions])

            # connect D1 to the output action layer
            for neuron in self.actions[nactions]:
                nest.ConvergentConnect(self.strD1[nactions], [neuron], weight=self.params['str_to_output_exc_w'], delay=self.params['str_to_output_exc_delay']) 

            if self.params['with_d2']: 
                for neuron in self.actions[nactions]:
                    nest.ConvergentConnect(self.strD2[nactions], [neuron], weight=self.params['str_to_output_inh_w'], delay=self.params['str_to_output_inh_delay'])	
            else:
                for tgt_action in xrange(self.params['n_actions']):
                    if nactions != tgt_action:
                        for neuron in self.actions[tgt_action]:
                            nest.ConvergentConnect(self.strD1[nactions],
                                    [neuron],
                                    weight=self.params['str_to_output_inh_w'],
                                    delay=self.params['str_to_output_inh_delay']) 

            if self.params['record_bg_volt']:
                nest.ConvergentConnect(self.voltmeter_action[nactions], self.actions[nactions])
                nest.RandomConvergentConnect(self.voltmeter_d1[nactions], self.strD1[nactions], int(self.params['random_connect_voltmeter']*self.params['num_msn_d1']))
                if self.params['with_d2']: 
                    nest.RandomConvergentConnect(self.voltmeter_d2[nactions], self.strD2[nactions], int(self.params['random_connect_voltmeter']*self.params['num_msn_d2']))

        # create supervisor
        if (self.params['training'] and self.params['supervised_on']):
            for nactions in xrange(self.params['n_actions']):
                self.supervisor[nactions] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_supervisor'], params=self.params['param_poisson_supervisor']  )
                for i in xrange(self.params['n_actions']):
                     if i == nactions:
                         nest.DivergentConnect(self.supervisor[nactions], self.strD1[i], weight=self.params['weight_supervisor_strd1'], delay=self.params['delay_supervisor_strd1'])
                     elif self.params['with_d2']: 
                         nest.DivergentConnect(self.supervisor[nactions], self.strD2[i], weight=self.params['weight_supervisor_strd2'], delay=self.params['delay_supervisor_strd2'])

            self.stop_supervisor()

        if self.params['training']:
            # create efference copy (for RBL, activates BOTH D1 and D2 actions at the same time)
            for nactions in xrange(self.params['n_actions']):
                self.efference_copy[nactions] = nest.Create('poisson_generator', self.params['num_neuron_poisson_efference'], params=self.params['param_poisson_efference'])
                self.efference_copy_d1[nactions] = nest.Create('poisson_generator', self.params['num_neuron_poisson_efference'], params=self.params['param_poisson_efference'])
                self.efference_copy_d2[nactions] = nest.Create('poisson_generator', self.params['num_neuron_poisson_efference'], params=self.params['param_poisson_efference'])
                nest.DivergentConnect(self.efference_copy[nactions], self.strD1[nactions], weight=self.params['weight_efference_strd1'], delay=self.params['delay_efference_strd1'])
                nest.DivergentConnect(self.efference_copy[nactions], self.strD2[nactions], weight=self.params['weight_efference_strd2'], delay=self.params['delay_efference_strd2'])

                nest.DivergentConnect(self.efference_copy_d1[nactions], self.strD1[nactions], weight=self.params['weight_efference_strd1'], delay=self.params['delay_efference_strd1'])
                nest.DivergentConnect(self.efference_copy_d2[nactions], self.strD2[nactions], weight=self.params['weight_efference_strd2'], delay=self.params['delay_efference_strd2'])

                nest.SetStatus(self.efference_copy[nactions], {'rate' : self.params['inactive_efference_rate']})

            self.stop_efference_copy()
         

        ###########################################################
        ############# Now we create bcpnn connections #############

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])
        if not self.params['are_MT_BG_connected']:
            self.create_input_pop()

        self.write_cell_gids_to_file()
        self.gids = {}
        self.gids['actions'] = self.get_cell_gids('actions')
        self.gids['d1'] = self.get_cell_gids('actions')
        self.gids['d2'] = self.get_cell_gids('actions')
#        print "BG model completed"


    def connect_d1_population(self):
        D1_conns = ''
        for i_ in xrange(self.params['n_actions']):
            src_pop = self.strD1[i_]
            for j_ in xrange(self.params['n_actions']):
                tgt_pop = self.strD1[j_]
                if self.params['synapse_d1_d1'] == 'bcpnn_synapse':
                    nest.SetDefaults(self.params['synapse_d1_d1'], params=self.params['params_synapse_d1_d1'])
                    nest.ConvergentConnect(src_pop, tgt_pop, model=self.params['synapse_d1_d1'])
                else:
                    nest.ConvergentConnect(src_pop, tgt_pop, self.params['w_d1_d1'], self.params['delay_d1_d1'])

    def connect_d2_population(self):
        for i_ in xrange(self.params['n_actions']):
            src_pop = self.strD2[i_]
            for j_ in xrange(self.params['n_actions']):
                tgt_pop = self.strD2[j_]
                nest.SetDefaults(self.params['synapse_d2_d2'], params=self.params['params_synapse_d2_d2'])
                nest.ConvergentConnect(src_pop, tgt_pop, model=self.params['synapse_d2_d2'])
#                else:
#                    nest.ConvergentConnect(src_pop, tgt_pop, self.params['w_d2_d2'], self.params['delay_d2_d2'])


    def connect_noise(self):

        self.noise_exc_d1 = {}
        self.noise_inh_d1 = {}
        self.noise_exc_actions = {}
        self.noise_inh_actions = {}
        for naction in xrange(self.params['n_actions']):
            self.noise_exc_d1[naction] = nest.Create('poisson_generator', self.params['num_msn_d1']) 
            self.noise_inh_d1[naction] = nest.Create('poisson_generator', self.params['num_msn_d1'])
            nest.SetStatus(self.noise_exc_d1[naction], {'rate': self.params['f_noise_exc_bg']})
            nest.SetStatus(self.noise_inh_d1[naction], {'rate': self.params['f_noise_inh_bg']})
            nest.Connect(self.noise_exc_d1[naction], self.strD1[naction], self.params['w_noise_exc_bg'], self.params['dt'])
            nest.Connect(self.noise_inh_d1[naction], self.strD1[naction], self.params['w_noise_inh_bg'], self.params['dt'])

            self.noise_exc_actions[naction] = nest.Create('poisson_generator', self.params['num_actions_output']) 
            self.noise_inh_actions[naction] = nest.Create('poisson_generator', self.params['num_actions_output'])
            nest.SetStatus(self.noise_exc_actions[naction], {'rate': self.params['f_noise_exc_bg']})
            nest.SetStatus(self.noise_inh_actions[naction], {'rate': self.params['f_noise_inh_bg']})
            nest.Connect(self.noise_exc_actions[naction], self.actions[naction], self.params['w_noise_exc_bg'], self.params['dt'])
            nest.Connect(self.noise_inh_actions[naction], self.actions[naction], self.params['w_noise_inh_bg'], self.params['dt'])

        if self.params['with_d2']:
            self.noise_exc_d2 = {}
            self.noise_inh_d2 = {}
            for naction in xrange(self.params['n_actions']):
                self.noise_exc_d2[naction] = nest.Create('poisson_generator', self.params['num_msn_d2']) 
                self.noise_inh_d2[naction] = nest.Create('poisson_generator', self.params['num_msn_d2'])
                nest.SetStatus(self.noise_exc_d2[naction], {'rate': self.params['f_noise_exc_bg']})
                nest.SetStatus(self.noise_inh_d2[naction], {'rate': self.params['f_noise_inh_bg']})
                nest.Connect(self.noise_exc_d2[naction], self.strD2[naction], self.params['w_noise_exc_bg'], self.params['dt'])
                nest.Connect(self.noise_inh_d2[naction], self.strD2[naction], self.params['w_noise_inh_bg'], self.params['dt'])




    def set_action_speed_mapping_bins(self):
        self.action_bins_x = []
        n_bins_x = np.int(np.round((self.params['n_actions'] - 1) / 2.))
        n_bins_y = np.int(np.round((self.params['n_actions'] - 1) / 2.))

#        if self.params['regular_tuning_prop']:
        v_scale_half = ((-1.) * np.logspace(np.log(self.params['v_min_out']) / np.log(self.params['log_scale']),
                            np.log(self.params['v_max_out']) / np.log(self.params['log_scale']), num=n_bins_x,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        v_scale_half.reverse()
        self.action_bins_x += v_scale_half
        self.action_bins_x += [0.]
        v_scale_half = (np.logspace(np.log(self.params['v_min_out']) / np.log(self.params['log_scale']),
                            np.log(self.params['v_max_out']) / np.log(self.params['log_scale']), num=n_bins_x,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        self.action_bins_x += v_scale_half
#        print 'BG: action_bins_x', self.action_bins_x


        ### the same for the y-direction
        self.action_bins_y = []
        v_scale_half = ((-1.) * np.logspace(np.log(self.params['v_min_out'])/np.log(self.params['log_scale']),
                            np.log(self.params['v_max_out'])/np.log(self.params['log_scale']), num=n_bins_y,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        v_scale_half.reverse()
        self.action_bins_y += v_scale_half
        self.action_bins_y += [0.]
        v_scale_half = (np.logspace(np.log(self.params['v_min_out']) / np.log(self.params['log_scale']),
                            np.log(self.params['v_max_out']) / np.log(self.params['log_scale']), num=n_bins_y,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        self.action_bins_y += v_scale_half
#        print 'BG: action_bins_y', self.action_bins_y

#        else:
#            self.action_bins_x = np.linspace(-self.params['v_max_out'], self.params['v_max_out'], n_bins_x)
#            self.action_bins_y = np.linspace(-self.params['v_max_out'], self.params['v_max_out'], n_bins_y)

        output_array = np.zeros((len(self.action_bins_x), 2))
#        header = '# first row: action_x, 2nd row: action_y'
        output_array[:, 0] = self.action_bins_x
        output_array[:, 1] = self.action_bins_y
        np.savetxt(self.params['bg_action_bins_fn'], output_array)#, header=header)



    def map_speed_to_action(self, speed, xy='x'):
        if xy == 'x':
            binning = self.action_bins_x
        else:
            binning = self.action_bins_y
        if speed > np.max(binning):
            action_index = self.params['n_actions'] - 1
        elif (speed < np.min(binning)):
            action_index = 0
        else:
            action_index = np.argmin(np.abs(speed - np.array(binning)))
#            cnt_u, bins = np.histogram(speed, binning)
#            action_index = cnt_u.nonzero()[0][0]

        if xy == 'x':
            print 'BG.map_speed_to_action (pc_id=%d, iteration=%d) : supervisor_speed=%.3f --> action: %d output_speed= %.3f' % (self.pc_id, self.iteration, speed, action_index, binning[action_index])

        return action_index



    def create_suboptimal_action_mapping(self):
        self.map_suboptimal_action = {}
        for action in xrange(self.params['n_actions']):
            rnd_action = int(action + self.params['suboptimal_training'] * utils.get_plus_minus(self.RNG))
            if rnd_action >= self.params['n_actions']:
                rnd_action = self.RNG.choice(xrange(self.params['n_actions'] - self.params['suboptimal_training']))
            elif rnd_action < 0:
                rnd_action = self.RNG.choice(xrange(0, self.params['suboptimal_training']))
            self.map_suboptimal_action[action] = rnd_action
#            self.map_suboptimal_action[action] = int(action + self.params['suboptimal_training'] * utils.get_plus_minus(self.RNG)) % self.params['n_actions']
        output_fn = self.params['bg_suboptimal_action_mapping_fn']
        output_file = file(output_fn, 'w')
        json.dump(self.map_suboptimal_action, output_file, indent=2)
        output_file.flush()
        output_file.close()


    def get_action_spike_based_memory_based(self, i_trial, stim_params):
        """
        Based on the spiking activity in the current iteration and the number of trials (i_trial) the stimulus has been presented, an action is selected.

        i_trial -- (int) indicating how often this stimulus has been presented (and is retrained)
        If i_trial == 0:
            the spiking activity determines the output action
            if there is no spiking activity, a random action is selected
        """

        if i_trial == 0:
            action = self.get_action() # updates self.t_iteration and self.t_current
            self.currently_trained_action = action[2] # update the currently trained action
            return action

        elif i_trial >= 1:
            all_outcomes = np.zeros(len(self.action_bins_x))
            for i_, action in enumerate(self.action_bins_x):    
                all_outcomes[i_] = utils.get_next_stim(self.params, stim_params, action)[0]
            best_action = np.argmin(np.abs(all_outcomes - .5))
            output_speed_x = self.action_bins_x[best_action]
            print 'BG for trial %d says (it %d, pc_id %d): do action %d, output_speed:' % (i_trial, self.t_current / self.params['t_iteration'], self.pc_id, best_action), output_speed_x
            self.t_current += self.params['t_iteration']
            self.advance_iteration()
            print 'DEBUG BG sets supervisor for action %d' % (best_action)
            for nactions in xrange(self.params['n_actions']):
                nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})
            nest.SetStatus(self.supervisor[best_action], {'rate' : self.params['active_supervisor_rate']})
            return (output_speed_x, 0, best_action)
             

        # TODO:
        # else: randomly choose another action use softmax_action_selection (without supervisor_state), b




    def get_optimal_action_for_stimulus(self, stim_params):
        action_bins = self.action_bins_x
        all_outcomes = np.zeros(len(action_bins))
        for i_, action in enumerate(action_bins):    
            all_outcomes[i_] = utils.get_next_stim(self.params, stim_params, action)[0]
        best_action_idx = np.argmin(np.abs(all_outcomes - .5))
        best_speed = self.action_bins_x[best_action_idx ]
        return (best_speed, 0, best_action_idx)


    def get_non_optimal_action_for_stimulus(self, stim_params):
        action_bins = self.action_bins_x
        all_outcomes = np.zeros(len(action_bins))
        for i_, action in enumerate(action_bins):    
            all_outcomes[i_] = utils.get_next_stim(self.params, stim_params, action)[0]
        best_action_idx = np.argmin(np.abs(all_outcomes - .5))
        if best_action_idx in self.all_action_idx:
            self.all_action_idx.remove(best_action_idx)
        non_optimal_action_idx = self.RNG.choice(self.all_action_idx)
        self.all_action_idx.remove(non_optimal_action_idx)
        speed = self.action_bins_x[non_optimal_action_idx]
        return (speed, 0, non_optimal_action_idx)


    def get_reward_from_action(self, chosen_action_idx, stim_params, training=False):
#        print 'debug get_reward_from_action chosen_action_idx:', chosen_action_idx , np.int(chosen_action_idx)
#        if training:
#            assert (chosen_action_idx == np.int(chosen_action_idx)), 'ERROR: get_reward_from_action requires an integer, i.e. the index of the action and NOT the speed it refers to! chosen_action_idx: %f' % chosen_action_idx
#        chosen_action_idx = np.int(chosen_action_idx)
        action_bins = self.action_bins_x
        all_outcomes = np.zeros(len(action_bins))
        for i_, action in enumerate(action_bins):    
            all_outcomes[i_] = utils.get_next_stim(self.params, stim_params, action)[0]
        best_action_idx = np.argmin(np.abs(all_outcomes - .5))
        # the reward is determined by the distance between the best_action and the chosen_action
#        best_speed = self.action_bins_x[best_action_idx ]
#        chosen_speed = self.action_bins_x[chosen_action_idx]

        reward = (self.params['K_max'] - self.params['shift_reward_distribution']) * np.exp( - float(chosen_action_idx - best_action_idx)**2 / (2. * self.params['sigma_reward_distribution'])) + self.params['shift_reward_distribution']

#        print 'debug get_reward_from_action: best_action_idx :', best_action_idx , 'best_speed', best_speed, 'chosen action idx', chosen_action_idx, 'chosen speed', chosen_speed, '\treward', reward
        return reward
#        if chosen_action != best_action_dx:
#            return -1.
#        else:
#            return 1.


    def softmax_action_selection(self, supervisor_state):
        """
        Will select the action corresponding to the supervisor_state 
        by applying softmax to all actions, i.e. depending on the temperature the 'correct' action
        is selected.
        For temperature = 0 all actions are equally likely (softmax yields a flat distribution), 
        and for temperature >= 10 the supervisor_state is basically mapped to the correct action (as in supervised_training).
        """
           
        (u, v) = supervisor_state 
        action_index_x = self.map_speed_to_action(u, xy='x')
        action_index_y = self.map_speed_to_action(v, xy='y')
        actions = np.zeros(self.params['n_actions'])
        actions[action_index_x] = 1.
        actions_softmax = utils.softmax(actions, self.params['softmax_temperature'])
        rnd_action = utils.draw_from_discrete_distribution(actions_softmax, size=1)[0]
           
        # set rate for all to inactive
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})
        nest.SetStatus(self.supervisor[rnd_action], {'rate' : self.params['active_supervisor_rate']})
        return (rnd_action, action_index_y)


    def supervised_training(self, supervisor_state):
        """
        Activates poisson generator of the required, teached, action and inactivates those of the nondesirable actions.
        The supervisor_state --- (u, v) is mapped to discretized states
        """
        print 'DEBUG supervised_training, supervisor_state:', supervisor_state
        (u, v) = supervisor_state 
        action_index_x = self.map_speed_to_action(u, xy='x') # would be interesting to test differences in x/y sensitivity here (as reported from Psychophysics)
        action_index_y = 0
#        action_index_y = self.map_speed_to_action(v, xy='y')

#        if self.params['suboptimal_training'] != 0.:
#            action_index_x = self.map_suboptimal_action[action_index_x]
#            action_index_y = self.map_suboptimal_action[action_index_y]
#            action_index_x += self.params['suboptimal_training'] * utils.plus_minus(self.RNG)
#            action_index_y += self.params['suboptimal_training'] * utils.plus_minus(self.RNG)

        print 'Debug BG iteration %d based on supervisor action choose action_index_x: %d ~ v_eye = %.2f, supervisor_state:' % (self.iteration, action_index_x, self.action_bins_x[action_index_x]), supervisor_state
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})
        nest.SetStatus(self.supervisor[action_index_x], {'rate' : self.params['active_supervisor_rate']})
        # TODO:  same for action_index_y
        return (action_index_x, action_index_y)
        

    def stop_supervisor(self):
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})


    def activate_efference_copy(self, it_0, it_1):
        print 'debug activity_memory ', self.iteration, self.activity_memory[it_0:it_1, :]
        recent_activity = self.activity_memory[it_0:it_1, :]
        mean_activity = np.zeros(self.params['n_actions'])
        for i_action in xrange(self.params['n_actions']):
            mean_activity[i_action] = recent_activity[:, i_action].mean()
        # set the activity for all to inactive
        for i_action in xrange(self.params['n_actions']):
            nest.SetStatus(self.efference_copy[i_action], {'rate' : self.params['inactive_efference_rate']})
        # set the activity for all to the recent mean activity
        for i_action in xrange(self.params['n_actions']):
            if mean_activity.max() != 0:
                amp = mean_activity[i_action] / mean_activity.max()
                nest.SetStatus(self.efference_copy[i_action], {'rate' : amp * self.params['active_efference_rate']})
            else:
                nest.SetStatus(self.efference_copy[i_action], {'rate' : 0.})


    def activate_efference_copy_d1_or_d2(self, action_idx, d1_or_d2):
        if d1_or_d2 == 'd1':
            for i_action in xrange(self.params['n_actions']):
                nest.SetStatus(self.efference_copy_d1[i_action], {'rate' : self.params['inactive_efference_rate']})
            nest.SetStatus(self.efference_copy_d1[action_idx], {'rate' : self.params['active_efference_rate']})
        else:
            for i_action in xrange(self.params['n_actions']):
                nest.SetStatus(self.efference_copy_d2[i_action], {'rate' : self.params['inactive_efference_rate']})
            nest.SetStatus(self.efference_copy_d2[action_idx], {'rate' : self.params['active_efference_rate']})


    def stop_efference_copy(self):
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.efference_copy[nactions], {'rate' : self.params['inactive_efference_rate']})

    def stop_supervisor(self):
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})




    def get_action(self, WTA=False, random_action=False):
        """
        Returns the selected action. Calls a selection function e.g. softmax, hardmax, ...
        random_action   -- is only relevant if no spikes are found in this iteration
                False: do nothing
                True:  choose a random action
                int:   choose to do this action instead
        """
        
        print 'BG.get_action ...'
        new_event_times = np.array([])
        new_event_gids = np.array([])
        t_new = self.t_current + self.params['t_iteration']
        for i_, recorder in enumerate(self.recorder_output.values()):
            all_events = nest.GetStatus(recorder)[0]['events']
            recent_event_idx = all_events['times'] > self.t_current
            if recent_event_idx.size > 0:
                new_event_times = np.r_[new_event_times, all_events['times'][recent_event_idx]]
                new_event_gids = np.r_[new_event_gids, all_events['senders'][recent_event_idx]]
            nest.SetStatus(recorder, [{'start': t_new}])

        if self.comm != None:
            gids_spiked, nspikes = utils.communicate_local_spikes(new_event_gids, self.comm)
        else:
            gids_spiked = np.unique(new_event_gids) - 1 # maybe here should be a - 1 (if there is one in communicate_local_spikes)
            nspikes = np.zeros(len(gids_spiked))
            for i_, gid in enumerate(gids_spiked):
                nspikes[i_] = (new_event_gids == gid + 1).nonzero()[0].size # + 1 because new_event gids holds the NEST gids, but there is a -1 in communicate_local_spikes 

        if len(nspikes) == 0:
            print 'No spikes found in iteration', self.t_current/self.params['t_iteration']
            self.t_current += self.params['t_iteration']
            self.advance_iteration()
            if random_action == False: # do nothing:
                return (0, 0, np.int(self.params['n_actions'] / 2)) # maye use 0 instead of np.nan
            elif random_action == True:
                rnd_action_idx = self.RNG.randint(0, self.params['n_actions'])
                output_speed_x = self.action_bins_x[rnd_action_idx]
                return (output_speed_x, 0, rnd_action_idx)
            else:
                assert (type(random_action) == type(0)), 'random_action is either True, False or an integer representing the action to be taken as default when action layer is silent'
                output_speed_x = self.action_bins_x[random_action]
                return (output_speed_x, 0, random_action)


        # switch between WTA behavior and Vector-Averaging
        if WTA:
            winning_nspikes = np.argmax(nspikes)
            winning_gid = gids_spiked[winning_nspikes]
#            print 'winning_gid', winning_gid
            winning_action = self.gid_to_action_via_spikerecorder[winning_gid+1]
            output_speed_x = self.action_bins_x[winning_action]
        else:
            vector_avg_action = 0.
            vector_avg_speed = 0.
            nspikes_sum = np.sum(nspikes)
            for i_, gid_ in enumerate(gids_spiked):
                action_idx = self.gid_to_action[int(gid_)]
                self.activity_memory[self.iteration, action_idx] += nspikes[i_]
                vector_avg_action += nspikes[i_] / float(nspikes_sum) * action_idx
                vector_avg_speed += nspikes[i_] / float(nspikes_sum) * self.action_bins_x[int(action_idx)]
            winning_action = vector_avg_action
            output_speed_x = vector_avg_speed

        print 'BG says (it %d, pc_id %d): do action %d, output_speed:' % (self.t_current / self.params['t_iteration'], self.pc_id, winning_action), output_speed_x
        self.t_current += self.params['t_iteration']

        self.advance_iteration()
        return (output_speed_x, 0, winning_action)


    def set_reward(self, rew):
    # absolute value of the reward
        if rew:
            nest.SetStatus(self.poisson_rew, {'rate' : self.params['active_poisson_rew_rate']})
            print 'debug: REWARD SET, activity is: ',nest.GetStatus(self.poisson_rew)[0]['rate'] 

    def stop_reward(self):
        nest.SetStatus(self.poisson_rew, {'rate' : self.params['inactive_poisson_rew_rate']})
        print 'debug: REWARD OFF, activity is: ', nest.GetStatus(self.poisson_rew)[0]['rate']


    def set_weights(self, src_pop, tgt_pop, conn_mat_ee, src_pop_idx, tgt_pop_idx):
       # set the connection weight after having loaded the conn_mat_ee
       nest.SetStatus(nest.GetConnections(src_pop, tgt_pop), {'weight': conn_mat_ee[src_pop_idx, tgt_pop_idx]})
       # nest.SetStatus(nest.FindConnections(src_pop, tgt_pop), {'weight': conn_mat_ee[src_pop_idx, tgt_pop_idx]})


    def set_bias(self, cell_type):
        # to be called from main_testing
        f = file(self.params['bias_%s_merged_fn' % cell_type], 'r')
        bias_values = json.load(f)
        if cell_type == 'd1':
            pop = self.strD1
        elif cell_type == 'd2':
            pop = self.strD2

        for gid in bias_values.keys(): 
            bias_value = bias_values[gid] * self.params['bias_gain']
            nest.SetStatus([int(gid)], {'I_e' : bias_value})


    def set_kappa_and_gain(self, source_gids, D1_or_D2, kappa, syn_gain, bias_gain):
        """
        source_gids -- is a list of gids, e.g. MotionPrediction.local_idx_exc
        D1_or_D2    -- is either a dictionary with int as key (=action) and gids as values, strD1 or strD2
        kappa       -- float
        syn_gain    -- float
        bias_gain   -- float
        """

        for i_action in xrange(self.params['n_actions']):
#            dummy = nest.GetConnections(source_gids, D1_or_D2[i_action])
#            print 'DEBUG dummy pc_id %d iteration %d action %d' % (self.pc_id, self.iteration, i_action), dummy, '\nDEBUG D1_or_D2 kappa=%.3f:' % kappa, D1_or_D2[i_action]
#            nest.SetStatus(dummy, {'K': float(kappa), 'gain': float(syn_gain)})

            nest.SetStatus(nest.GetConnections(source_gids, D1_or_D2[i_action]), {'K': float(kappa), 'gain': float(syn_gain), 't_k': nest.GetKernelStatus()['time']})

            # verify that kappa is now non zero
#            conns = nest.GetConnections(source_gids, D1_or_D2[i_action]) 
#            for c in conns:
#                cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
#                K_check = cp[0]['K']
#                print 'DEBUG check K:', K_check, c[0], c[1]


        for i_action in xrange(self.params['n_actions']):
            nest.SetStatus(D1_or_D2[i_action], {'K': float(kappa), 'gain': float(bias_gain)})


    def get_cell_gids(self, cell_type):
        cell_gids = []
        if cell_type == 'd1':
            for nactions in range(self.params['n_actions']):
                cell_gids.append(self.strD1[nactions])
        elif cell_type == 'd2' and self.params['with_d2']:
            for nactions in range(self.params['n_actions']):
                cell_gids.append(self.strD2[nactions])
        elif cell_type == 'actions':
            for nactions in range(self.params['n_actions']):
                cell_gids.append(self.actions[nactions])
        return cell_gids


    def write_cell_gids_to_file(self):
        d = {}
        for cell_type in self.params['bg_cell_types']:
            d[cell_type] = self.get_cell_gids(cell_type)
        d['gid_to_action_D1'] = self.gid_to_action_D1
        d['gid_to_action_D2'] = self.gid_to_action_D2
        d['gid_to_action'] = self.gid_to_action
        output_fn = self.params['bg_gids_fn']
        print 'Writing cell_gids to:', output_fn
        f = file(output_fn, 'w')
        json.dump(d, f, indent=2)
        return d


    def advance_iteration(self):
        self.iteration += 1


    def get_eye_direction(self):
        """
        Returns the efference copy, i.e. an internal copy of an outgoing (movement) signal.
        """
        pass
    

    def move_eye(self):
        """
        Select an action based on the current state and policy
        update the state
        """
        pass
