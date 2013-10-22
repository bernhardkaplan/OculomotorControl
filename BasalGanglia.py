import numpy as np
import nest
import utils



class BasalGanglia(object):

    def __init__(self, params, comm=None):

        self.params = params
        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.comm = comm # mpi communicator needed to broadcast nspikes between processes
        if comm != None:
            assert (comm.rank == self.pc_id), 'mpi4py and NEST tell me different PIDs!'
            assert (comm.size == self.n_proc), 'mpi4py and NEST tell me different PIDs!'

        self.strD1 = {}
        self.strD2 = {}
        self.actions = {}
        self.rp = {}
        self.recorder_output= {}
        self.recorder_output_gidkey = {}

        self.t_current = 0 
        self.voltmeter_action = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
        nest.SetStatus(self.voltmeter_action,[{"to_file": True, "withtime": True, 'label' : self.params['bg_action_volt_fn']}])

        #Creates D1 and D2 populations in STRIATUM, connections are created later
        for nactions in range(self.params['n_actions']):
            self.strD1[nactions] = nest.Create(self.params['model_exc_neuron'], self.params['num_msn_d1'], params= self.params['param_msn_d1'])
            self.strD2[nactions] = nest.Create(self.params['model_inh_neuron'], self.params['num_msn_d2'], params= self.params['param_msn_d2'])

        # Creates the different Populations, STR_D1, STR_D2 and Actions, and then create the Connections
        for nactions in range(self.params['n_actions']):
            self.actions[nactions] = nest.Create(self.params['model_bg_output_neuron'], self.params['num_actions_output'], params= self.params['param_bg_output'])
            self.recorder_output[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_output_action'])
            self.recorder_output_gidkey[self.recorder_output[nactions][0]] = nactions
            nest.SetStatus(self.recorder_output[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['bg_spikes_fn']}])
            nest.ConvergentConnect(self.actions[nactions], self.recorder_output[nactions])
            for neuron in self.actions[nactions]:
                nest.ConvergentConnect(self.strD1[nactions], [neuron], weight=self.params['str_to_output_exc_w'], delay=self.params['str_to_output_exc_delay']) 
                nest.ConvergentConnect(self.strD2[nactions], [neuron], weight=self.params['str_to_output_inh_w'], delay=self.params['str_to_output_inh_delay'])	

            nest.ConvergentConnect(self.voltmeter_action, self.actions[nactions])

        # Used to create external poisson process to make the system learns the desired output action by stimulating its relative striatal D1 population and the striatal D2 populations associated with the other actions.
        if self.params['supervised_on']:
            self.supervisor = {}
            for nactions in range(self.params['n_actions']):
                self.supervisor[nactions] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_supervisor'], params = self.params['param_poisson_supervisor']  )
                for supervisor_neuron in self.supervisor[nactions]:
                    for i in range(self.params['n_actions']):
                        if i != nactions:
                            nest.DivergentConnect([supervisor_neuron], self.strD2[i], weight=self.params['weight_supervisor_strd2'], delay=self.params['delay_supervisor_strd2'])
                    nest.DivergentConnect([supervisor_neuron], self.strD1[nactions], weight=self.params['weight_supervisor_strd1'], delay=self.params['delay_supervisor_strd1'])


        else: 
        # Creates the reward population and its poisson input and the RP population and then connects theses different populations.
            self.rew = nest.Create( self.params['model_rew_neuron'], self.params['num_rew_neurons'], params= self.params['param_rew_neuron'] )
            for index_rp in range(self.params['n_actions'] * self.params['n_states']):
                self.rp[index_rp] = nest.Create(self.params['model_rp_neuron'], self.params['num_rp_neurons'], params= self.params['param_rp_neuron'] )
            for nron_rew in self.rew:
                nest.ConvergentConnect( self.rp[index_rp], [nron_rew], weight=self.params['weight_rp_rew'], delay=self.params['delay_rp_rew']  )
            self.poisson_rew = nest.Create( self.params['model_poisson_rew'], self.params['num_poisson_rew'], params=self.params['param_poisson_rew'] )
            for nron_poisson in self.poisson_rew :
                nest.DivergentConnect([nron_poisson], self.rew, weight=self.params['weight_poisson_rew'], delay=self.params['delay_poisson_rew'])


            # Connects reward population back to the STR D1 and D2 populations, and to the RP, to inform them about the outcome (positive or negative compared to the prediction)
            for neur_rew in self.rew:
                for i_action in range(self.params['n_actions']):
                    nest.DivergentConnect([neur_rew], self.strD1[i_action], weight=self.params['weight_rew_strD1'], delay=self.params['delay_rew_strD1'] )
                    nest.DivergentConnect([neur_rew], self.strD2[i_action], weight=self.params['weight_rew_strD2'], delay=self.params['delay_rew_strD2'] )
                for i_rp in range(self.params['n_states'] * self.params['n_actions']):
                    nest.DivergentConnect([neur_rew], self.rp[i_rp], weight=self.params['weight_rew_rp'], delay=self.params['delay_rew_rp'] )




        ###########################################################
        ############# Now we create bcpnn connections #############

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])

        if not self.params['are_MT_BG_connected']:
            self.create_input_pop()

        if not self.params['supervised_on']:
        # Creates RP populations and the connections from states and actions to the corresponding RP populations
            modulo_i = 0
            for index_rp in range(self.params['n_actions'] * self.params['n_states']):
                nest.SetDefaults( self.params['bcpnn'], params= self.params['param_actions_rp'])
                for naction in self.actions[modulo_i % self.params['n_states']]:
                    nest.DivergentConnect([naction], self.rp[index_rp], model=self.params['actions_rp'])
                nest.SetDefaults(self.params['bcpnn'], params=self.params['param_states_rp'])
                for nstate in self.states[int( modulo_i / self.params['n_actions'] ) ]:
                    nest.DivergentConnect([nstate], self.rp[index_rp], model=self.params['states_rp'])   
                modulo_i += 1

        print "BG model completed"


    # used as long as MT and BG are not directly connected
    def create_input_pop(self):
        """
        Creates the input states populations, and their respective poisson pop,  and connect them to Striatum msns D1 D2 populations
        """
        self.states = {}
        self.input_poisson = {}

        for nstates in range(self.params['n_states']):
            self.input_poisson[nstates] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_input_BG'], params = self.params['param_poisson_pop_input_BG']  )
            self.states[nstates] = nest.Create( self.params['model_exc_neuron'], self.params['num_neuron_states'], params = self.params['param_states_pop']  )
            for neuron_poisson in self.input_poisson[nstates]:
                nest.DivergentConnect([neuron_poisson], self.states[nstates], weight=self.params['weight_poisson_input'], delay=self.params['delay_poisson_input'])
            for neuron_input in self.states[nstates]:
                for nactions in range(self.params['n_actions']):
                    nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d2_MT_BG'])
                    nest.DivergentConnect([neuron_input], self.strD1[nactions], model=self.params['synapse_d1_MT_BG'])
                    nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d2_MT_BG'])
                    nest.DivergentConnect([neuron_input], self.strD2[nactions], model=self.params['synapse_d2_MT_BG'])

        print "BG input stage created"

        pass


    def supervised_training(self, supervisor_state):
        """
        Activates poisson generator of the required, teached, action and inactivates the one of the nondesirable actions.
        """
        (x, y, u, v) = supervisor_state 
        action = [0, 0]
        action[0] = (x - .5) + u * self.params['t_iteration'] / self.params['t_cross_visual_field']
        action[1] = (y - .5) + v * self.params['t_iteration'] / self.params['t_cross_visual_field']

        print 'debug supervisor_state', supervisor_state
        print 'debug supervisor action', action

        # select an action based on the supervisor state information
        action_bins_x = np.linspace(-self.params['v_max_tp'], self.params['v_max_tp'], self.params['n_actions'])
        cnt_u, bins = np.histogram(action[0], action_bins_x)
        action_index_x = cnt_u.nonzero()[0][0]

        action_bins_y = np.linspace(-self.params['v_max_tp'], self.params['v_max_tp'], self.params['n_actions'])
        cnt_v, bins = np.histogram(action[1], action_bins_y)
        action_index_y = cnt_v.nonzero()[0][0]

#        print 'debug action_index', type(action_index)
        for nactions in range(self.params['n_actions']):
            nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})
        nest.SetStatus(self.supervisor[action_index_x], {'rate' : self.params['active_supervisor_rate']})
        # TODO:  same for action_index_y
        


    def set_state(self, state):
        """
        Informs BG about the current state. Used only when input state is internal to BG. Poisson population stimulated.
        """
        for i in range(self.params['n_states']):
            nest.SetStatus(self.input_poisson[i], {'rate' : self.params['inactive_poisson_input_rate']})
        nest.SetStatus(self.input_poisson[state], {'rate' : self.params['active_poisson_input_rate']})

        pass


    def get_action(self, state):
        """
        Returns the selected action. Calls a selection function e.g. softmax, hardmax, ...
        state -- is (x, y, v_x, v_y) vector-averaging based on MPN spiking activity
        """
#	action = softmax(self.actions)
#nothing is computed from BG, here it returns a random int between 0 and n_actions.

        # get the spikes from the network
#        action = np.random.randint(self.params['n_actions'])
#        self.recorder_output
#        for i_, a in enumerate([self.recorder_output]):


        
        new_event_times = np.array([])
        new_event_gids = np.array([])
        for i_, recorder in enumerate(self.recorder_output.values()):
            all_events = nest.GetStatus(recorder)[0]['events']
            recent_event_idx = all_events['times'] > self.t_current
            
            if recent_event_idx.size > 0:
                new_event_times = np.r_[new_event_times, all_events['times'][recent_event_idx].tolist()]
                new_event_gids = np.r_[new_event_gids, all_events['senders'][recent_event_idx].tolist()]

        if self.comm != None:
            gids_spiked, nspikes = utils.communicate_local_spikes(new_event_gids, self.comm)
        else:
            gids_spiked = new_event_gids.unique() - 1
            nspikes = np.zeros(len(new_event_gids))
            for i_, gid in enumerate(new_event_gids):
                nspikes[i_] = (new_event_gids == gid).nonzero()[0].size
        winning_nspikes = np.argmax(nspikes)
        winning_gid = gids_spiked[winning_nspikes]
        winning_action = self.recorder_output_gidkey[winning_gid]
        output_speed = self.translate_action_to_speed(winning_action) 
        print 'BG says: do action', winning_action, output_speed
        self.t_current += self.params['t_iteration']

        return output_speed


    def translate_action_to_speed(self, action_idx):

        action_bins = np.linspace(-self.params['v_max_tp'], self.params['v_max_tp'], self.params['n_actions'])
        output_speed = action_bins[action_idx]
        return (output_speed, 0)


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
