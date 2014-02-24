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
        self.efference_copy = {}


        # Recording devices
        self.recorder_output= {}
        self.recorder_output_gidkey = {}
        self.recorder_d1 = {}
        self.recorder_d2 = {}
        self.recorder_states = {}
        self.recorder_efference = {}
        self.recorder_rp = {}
        self.recorder_rew = nest.Create("spike_detector", params= self.params['spike_detector_rew'])
        nest.SetStatus(self.recorder_rew,[{"to_file": True, "withtime": True, 'label' : self.params['rew_spikes_fn']}])

        #self.recorder_test_rp = nest.Create("spike_detector", params= self.params['spike_detector_test_rp'])
        #nest.SetStatus(self.recorder_test_rp, [{"to_file": True, "withtime": True, 'label' : self.params['test_rp_spikes_fn']}])

        self.voltmeter_rp = {}
        self.voltmeter_rp = {}
        self.voltmeter_rew = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
        nest.SetStatus(self.voltmeter_rew, [{"to_file": True, "withtime": True, 'label' : self.params['rew_volt_fn']}])
        self.voltmeter_action = {}
        self.voltmeter_d1 = {}
        self.voltmeter_d2 = {}
        

        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder'], 'overwrite_files': True})

        self.t_current = 0 
        

        #Creates D1 and D2 populations in STRIATUM, connections are created later
        for nactions in range(self.params['n_actions']):
            self.strD1[nactions] = nest.Create(self.params['model_exc_neuron'], self.params['num_msn_d1'], params= self.params['param_msn_d1'])
            self.strD2[nactions] = nest.Create(self.params['model_inh_neuron'], self.params['num_msn_d2'], params= self.params['param_msn_d2'])
            
            self.voltmeter_d1[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
            nest.SetStatus(self.voltmeter_d1[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d1_volt_fn']+ str(nactions)}])
            self.voltmeter_d2[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
            nest.SetStatus(self.voltmeter_d2[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d2_volt_fn']+ str(nactions)}])

        # Creates the output ACTIONS populations, and then create the Connections with STR
        for nactions in range(self.params['n_actions']):
            self.actions[nactions] = nest.Create(self.params['model_bg_output_neuron'], self.params['num_actions_output'], params= self.params['param_bg_output'])
            
            self.voltmeter_action[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
            nest.SetStatus(self.voltmeter_action[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['actions_volt_fn']+ str(nactions)}])
            self.recorder_output[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_action'])
            self.recorder_d1[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_d1'])
            self.recorder_d2[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_d2'])
            for ind in xrange(self.params['num_actions_output']):
                self.recorder_output_gidkey[self.actions[nactions][ind]] = nactions
            nest.SetStatus(self.recorder_output[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['actions_spikes_fn']+ str(nactions)}])
            nest.SetStatus(self.recorder_d1[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d1_spikes_fn']+ str(nactions)}])
            nest.SetStatus(self.recorder_d2[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['d2_spikes_fn']+ str(nactions)}])
            nest.ConvergentConnect(self.actions[nactions], self.recorder_output[nactions])
            nest.ConvergentConnect(self.strD1[nactions], self.recorder_d1[nactions])
            nest.ConvergentConnect(self.strD2[nactions], self.recorder_d2[nactions])
            for neuron in self.actions[nactions]:
                nest.ConvergentConnect(self.strD1[nactions], [neuron], weight=self.params['str_to_output_exc_w'], delay=self.params['str_to_output_exc_delay']) 
                nest.ConvergentConnect(self.strD2[nactions], [neuron], weight=self.params['str_to_output_inh_w'], delay=self.params['str_to_output_inh_delay'])	


            nest.ConvergentConnect(self.voltmeter_action[nactions], self.actions[nactions])
            nest.RandomConvergentConnect(self.voltmeter_d1[nactions], self.strD1[nactions], int(self.params['random_connect_voltmeter']*self.params['num_msn_d1']))
            nest.RandomConvergentConnect(self.voltmeter_d2[nactions], self.strD2[nactions], int(self.params['random_connect_voltmeter']*self.params['num_msn_d2']))

        print 'debug recorder gid', self.recorder_output_gidkey

        # Creates and connects the EFFERENCE COPY population. This actives the D1 population coding for the selected action and the D2 populations of non-selected actions, in STR
        for nactions in xrange(self.params['n_actions']):
            self.efference_copy[nactions] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_efference'], params = self.params['param_poisson_efference']  )
            for i in xrange(self.params['n_actions']):
                 if i != nactions:
                     nest.RandomDivergentConnect(self.efference_copy[nactions], self.strD2[i],int(self.params['random_divconnect_poisson']*self.params['num_neuron_poisson_efference']), weight=self.params['weight_efference_strd2'], delay=self.params['delay_efference_strd2'])
                 else:
                     nest.RandomDivergentConnect(self.efference_copy[nactions], self.strD1[i], int(self.params['random_divconnect_poisson']*self.params['num_neuron_poisson_efference']), weight=self.params['weight_efference_strd1'], delay=self.params['delay_efference_strd1'])
            
            
            self.recorder_efference[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_efference'])
            nest.SetStatus(self.recorder_efference[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['efference_spikes_fn']+ str(nactions)}])
            nest.ConvergentConnect(self.efference_copy[nactions], self.recorder_efference[nactions])


         
        # Creates the REWARD population and its poisson input and the RP population and then connects theses different populations.
        self.rew = nest.Create( self.params['model_rew_neuron'], self.params['num_rew_neurons'], params= self.params['param_rew_neuron'] )
        for index_rp in range(self.params['n_actions'] * self.params['n_states']):
            self.rp[index_rp] = nest.Create(self.params['model_rp_neuron'], self.params['num_rp_neurons'], params= self.params['param_rp_neuron'] )
            for nron_rp in self.rp[index_rp]:
                nest.DivergentConnect( [nron_rp], self.rew, weight=self.params['weight_rp_rew'], delay=self.params['delay_rp_rew']  )
            #nest.ConvergentConnect(self.recorder_test_rp, self.rp[index_rp])
            self.recorder_rp[index_rp] = nest.Create("spike_detector", params= self.params['spike_detector_rp'])
            nest.SetStatus(self.recorder_rp[index_rp],[{"to_file": True, "withtime": True, 'label' : self.params['rp_spikes_fn']+ str(index_rp)}])
            nest.ConvergentConnect(self.rp[index_rp],self.recorder_rp[index_rp])
            self.voltmeter_rp[index_rp] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
            nest.SetStatus(self.voltmeter_rp[index_rp], [{"to_file": True, "withtime": True, 'label' : self.params['rp_volt_fn']+ str(index_rp)}])
            nest.ConvergentConnect(self.voltmeter_rp[index_rp], self.rp[index_rp])

        self.poisson_rew = nest.Create( self.params['model_poisson_rew'], self.params['num_poisson_rew'], params=self.params['param_poisson_rew'] )
        nest.RandomDivergentConnect(self.poisson_rew, self.rew, int(self.params['random_divconnect_poisson']*self.params['num_poisson_rew']), weight=self.params['weight_poisson_rew'], delay=self.params['delay_poisson_rew'])


        # Connects reward population back to the STR D1 and D2 populations, and to the RP, to inform them about the outcome (positive or negative compared to the prediction)
        for neur_rew in self.rew:
            for i_action in range(self.params['n_actions']):
                nest.DivergentConnect([neur_rew], self.strD1[i_action], weight=self.params['weight_rew_strD1'], delay=self.params['delay_rew_strD1'] )
                nest.DivergentConnect([neur_rew], self.strD2[i_action], weight=self.params['weight_rew_strD2'], delay=self.params['delay_rew_strD2'] )
            for i_rp in range(self.params['n_states'] * self.params['n_actions']):
                nest.DivergentConnect([neur_rew], self.rp[i_rp], weight=self.params['weight_rew_rp'], delay=self.params['delay_rew_rp'] )
        nest.ConvergentConnect(self.rew, self.recorder_rew)
        nest.ConvergentConnect(self.voltmeter_rew, self.rew)

        #self.create_brainstem()


        ###########################################################
        ############# Now we create BCPNN connections #############

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])
        self.create_input_pop()
        #self.create_bcpnn_sensorimotor()
       
        # Creates RP populations and the connections from states and actions to the corresponding RP populations
        for istate in range(self.params['n_states']):
            for iaction in range(self.params['n_actions']):
                nest.SetDefaults( self.params['bcpnn'], params= self.params['param_actions_rp'])
                for neuron_a in self.actions[iaction]:
                    nest.DivergentConnect([neuron_a], self.rp[iaction+istate*self.params['n_actions']], model=self.params['actions_rp'])
                
                nest.SetDefaults( self.params['bcpnn'], params= self.params['param_states_rp'])
                for neuron_s in self.states[istate]:
                    nest.DivergentConnect([neuron_s], self.rp[iaction + istate*self.params['n_actions']], model=self.params['states_rp'] )

        print "BG model completed"


    def create_input_pop(self):
        """
        Creates the inputs populations, and their respective poisson pop, and connect them to Striatum MSNs D1 D2 populations
        """
        self.states = {}
        self.input_poisson = {}

        for nstates in range(self.params['n_states']):
            self.input_poisson[nstates] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_input_BG'], params = self.params['param_poisson_pop_input_BG']  )
            self.states[nstates] = nest.Create( self.params['model_exc_neuron'], self.params['num_neuron_states'], params = self.params['param_states_pop']  )
            nest.RandomDivergentConnect(self.input_poisson[nstates], self.states[nstates],int(self.params['random_divconnect_poisson']*self.params['num_neuron_poisson_input_BG']), weight=self.params['weight_poisson_input'], delay=self.params['delay_poisson_input'])
            for neuron_input in self.states[nstates]:
                for nactions in range(self.params['n_actions']):
                    nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d1'])
                    nest.DivergentConnect([neuron_input], self.strD1[nactions], model=self.params['synapse_d1'])
                    nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d2'])
                    nest.DivergentConnect([neuron_input], self.strD2[nactions], model=self.params['synapse_d2'])
            
            self.recorder_states[nstates] = nest.Create("spike_detector", params= self.params['spike_detector_states'])
            nest.SetStatus(self.recorder_states[nstates],[{"to_file": True, "withtime": True, 'label' : self.params['states_spikes_fn'] + str(nstates)}])
            nest.ConvergentConnect(self.states[nstates], self.recorder_states[nstates])

        print "BG input stage created"

    def create_brainstem(self):
        """
        Creates a new output population (brainstem)  and a static connection between actions and output. 
        """
        self.brainstem = {}
        self.recorder_brainstem = {}
        for i in xrange(self.params['n_actions']):
            self.brainstem[i] = nest.Create( self.params['model_brainstem_neuron'], self.params['num_brainstem_neurons'], params= self.params['param_brainstem_neuron'] )
            self.recorder_brainstem[i] = nest.Create("spike_detector", params= self.params['spike_detector_brainstem'])
            nest.SetStatus(self.recorder_brainstem[i],[{"to_file": True, "withtime": True, 'label' : self.params['brainstem_spikes_fn'] + str(i)}])
            nest.ConvergentConnect(self.brainstem[i], self.recorder_brainstem[i])
            for neur in self.brainstem[i]:
                nest.ConvergentConnect(self.actions[i], [neur], model=self.params['action_brainstem_synapse'])
 
        print "Brainstem output created"


    def create_bcpnn_sensorimotor(self):
        """
        Creates a plastic connection from state populations to this output population
        """
        nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_states_brainstem'])
        for ns in xrange(self.params['n_states']):
            for neur in self.states[ns]:
                nest.DivergentConnect([neur], self.brainstem[ns % self.params['n_states']], model=self.params['synapse_states_brainstem'] )

        print "Sensorimotor connection completed"


    def set_efference_copy(self, action):
        """
        Activates poisson generator to activate the selected action accordingly in the different pathways accordingly to the complementary activity.
        """

        print 'debug efference copy action', action

        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.efference_copy[nactions], {'rate' : self.params['inactive_efference_rate']})
            print 'debug: EFFERENCE OFF for ACTION', nactions  , 'activity is: ',nest.GetStatus(self.efference_copy[nactions])[0]['rate'] 
        nest.SetStatus(self.efference_copy[action], {'rate' : self.params['active_full_efference_rate']})
        print 'debug: EFFERENCE SET for ACTION',action ,' activity is: ',nest.GetStatus(self.efference_copy[action])[0]['rate'] 
        
    def stop_efference(self):
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.efference_copy[nactions], {'rate' : self.params['inactive_efference_rate']})
            print 'debug: EFFERENCE OFF, activity is: ',nest.GetStatus(self.efference_copy[nactions])[0]['rate'] 



    def set_state(self, state):
        """
        Informs BG about the current state. Used only when input state is internal to BG. Poisson population stimulated.
        """
        for i in range(self.params['n_states']):
            nest.SetStatus(self.input_poisson[i], {'rate' : self.params['inactive_poisson_input_rate']})
            print 'debug: STATE OFF, activity is: ',nest.GetStatus(self.input_poisson[i])[0]['rate'] 
        nest.SetStatus(self.input_poisson[state], {'rate' : self.params['active_poisson_input_rate']})
        print 'debug: STATE ON, activity is: ',nest.GetStatus(self.input_poisson[state])[0]['rate'] 

    def stop_state(self):
        """
        Stops poisson input to BG, no current state.
        """
        for i in range(self.params['n_states']):
            nest.SetStatus(self.input_poisson[i], {'rate' : self.params['inactive_poisson_input_rate']})
            print 'debug: STATE OFF, activity is: ',nest.GetStatus(self.input_poisson[i])[0]['rate'] 

    def set_rest(self):
        """
        Informs BG about the current state. Used only when input state is internal to BG. Poisson population stimulated.
        """
        self.stop_state()
        self.stop_reward()
        self.stop_efference()

    def get_action(self):
        """
        Returns the selected action. Calls a selection function e.g. softmax, hardmax, ...
        """
        
        new_event_times = np.array([])
        new_event_gids = np.array([])
        for i_, recorder in enumerate(self.recorder_output.values()):
            all_events = nest.GetStatus(recorder)[0]['events']
            recent_event_idx = all_events['times'] > self.t_current
            if recent_event_idx.size > 0:
                new_event_times = np.r_[new_event_times, all_events['times'][recent_event_idx]]
                new_event_gids = np.r_[new_event_gids, all_events['senders'][recent_event_idx]]

        if self.comm != None:
            gids_spiked, nspikes = utils.communicate_local_spikes(new_event_gids, self.comm)
        else:
            gids_spiked = new_event_gids.unique() - 1
            nspikes = np.zeros(len(new_event_gids))
            for i_, gid in enumerate(new_event_gids):
                nspikes[i_] = (new_event_gids == gid).nonzero()[0].size
        if sum(nspikes)==0:
            print '*******no spikes*******'
            winning_action = utils.communicate_action(self.comm, self.params['n_actions'])
            
        else:    
            winning_nspikes = np.argmax(nspikes)
            winning_gid = gids_spiked[winning_nspikes]
            print 'winning gid: ', winning_gid
            winning_action = self.recorder_output_gidkey[winning_gid+1]
        
        print 'BG says (it %d, pc_id %d): do action %d' % (self.t_current / self.params['t_iteration'], self.pc_id, winning_action)
        print 'Activity is ', len(new_event_times)
        self.t_current += self.params['t_iteration']

        return (int(winning_action))


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

    def set_gain(self, gain):
        # implement option to change locally to d1 or d2 or RP

       for nstate in range(self.params['n_states']):
           for naction in range(self.params['n_actions']):
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction]), {'gain':gain})
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD2[naction]), {'gain':gain})
    	
       for index_rp in range(self.params['n_actions'] * self.params['n_states']):
            for naction in range(self.params['n_actions']):
               nest.SetStatus(nest.GetConnections(self.actions[naction], self.rp[index_rp % self.params['n_states']]), {'gain':gain})
            for nstate in range(self.params['n_states']):
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.rp[int(index_rp / self.params['n_actions'])]), {'gain':gain})
        
#       for nstates in range(self.params['n_states']):
#           #            print 'getstatus ' , nest.GetStatus(nest.FindConnections(self.states[nstates]))
#           nest.SetStatus([nest.FindConnections(self.states[nstates])], {'gain':gain})
#
#       for index_rp in range(self.params['n_actions']) :
#           nest.SetStatus([nest.FindConnections(self.actions[nactions])], {'gain':gain})

    def set_kappa_ON(self, k, state, action):
        # implement option to change locally to d1 or d2 or RP
        #To implement the opposite effect on the D1 and D2 MSNs of the dopamine release, -k is sent to D2
       k1 = k
       k2 = -k
     #  if k1 < 0:
     #      k1 = k1 / self.params['n_actions']
     #      #k1 = 0.
     #  if k2 < 0:
     #      k2 = k2 / self.params['n_actions'] 
     #      #k2 = 0.
     #  for nstate in range(self.params['n_states']):
     #      for naction in range(self.params['n_actions']):
     #          nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction]), {'K':k1})
     #          nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD2[naction]), {'K':k2})
     #      print 'KAPPA D1 SET TO', k1, 'D2 to', k2,'for STATE ', nstate
     #  for nact in xrange(self.params['n_actions']):
     #      nest.SetStatus(self.strD1[nact], {'kappa':k1} )
     #      nest.SetStatus(self.strD2[nact], {'kappa':k2} )

       nest.SetStatus(nest.GetConnections(self.states[state], self.strD1[action]), {'K': k} )
       nest.SetStatus(self.strD1[action], {'kappa': k})
       for naction in range(self.params['n_actions']):
           # if naction == action:
               nest.SetStatus(nest.GetConnections(self.states[state], self.strD2[naction]), {'K':-k})
               nest.SetStatus(self.strD2[naction], {'kappa': -k})
          # else:
               #         nest.SetStatus(nest.GetConnections(self.states[state], self.strD2[naction]), {'K':k})
     #          nest.SetStatus(self.strD2[naction], {'kappa': k})
     #          nest.SetStatus(nest.GetConnections(self.states[state], self.strD1[naction]), {'K':-k})
     #          nest.SetStatus(self.strD1[naction], {'kappa': -k})
       
       nest.SetStatus(nest.GetConnections(self.states[state], self.rp[action+state*self.params['n_actions']]), {'K':k})
       nest.SetStatus(nest.GetConnections(self.actions[action], self.rp[action+state*self.params['n_actions']]), {'K':k})
       nest.SetStatus(self.rp[state+action*self.params['n_states']], {'kappa':k} )


     #  for index_rp in range(self.params['n_actions'] * self.params['n_states']):
     #       for naction in range(self.params['n_actions']):
     #          nest.SetStatus(nest.GetConnections(self.actions[naction], self.rp[index_rp % self.params['n_states']]), {'K':k})
     #       for nstate in range(self.params['n_states']):
     #   		nest.SetStatus(nest.GetConnections(self.states[nstate], self.rp[int(index_rp / self.params['n_actions'])]), {'K':k})
     #       nest.SetStatus(self.rp[index_rp], {'kappa':k} )

#       for nstates in range(self.params['n_states']):
#           nest.SetStatus(nest.FindConnections([self.states[nstates]]), {'K':k})
#
#       for index_rp in range(self.params['n_actions']) :
#           nest.SetStatus(nest.FindConnections([self.actions[nactions]]), {'K':k})

    def set_kappa_OFF(self):
        # implement option to change locally to d1 or d2 or RP
        #To implement the opposite effect on the D1 and D2 MSNs of the dopamine release, -k is sent to D2
       for nstate in range(self.params['n_states']):
           for naction in range(self.params['n_actions']):
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD1[naction]), {'K':0.})
               nest.SetStatus(nest.GetConnections(self.states[nstate], self.strD2[naction]), {'K':0.})
       for nact in xrange(self.params['n_actions']):
           nest.SetStatus(self.strD1[nact], {'kappa':0.} )
           nest.SetStatus(self.strD2[nact], {'kappa':0.} )
       #nest.SetStatus(nest.GetConnections(self.states[state], self.strD1[action]), {'K': 0.} )
       #nest.SetStatus(self.strD1[action], {'kappa': 0.})



       for index_rp in range(self.params['n_actions'] * self.params['n_states']):
            for naction in range(self.params['n_actions']):
               nest.SetStatus(nest.GetConnections(self.actions[naction], self.rp[index_rp % self.params['n_states']]), {'K':0.})
            for nstate in range(self.params['n_states']):
    			nest.SetStatus(nest.GetConnections(self.states[nstate], self.rp[int(index_rp / self.params['n_actions'])]), {'K':0.})
            nest.SetStatus(self.rp[index_rp], {'kappa':0.} )




    def load_weights(self, training_params):
        """
        Connects the sensor layer (motion-prediction network, MPN) to the Basal Ganglia 
        based on the weights found in conn_folder
        """
        print 'debug', os.path.exists(training_params['d1_weights_fn'])
        print 'debug', training_params['d1_weights_fn']
        if not os.path.exists(training_params['d1_weights_fn']):
            # merge the connection files
            merge_pattern = training_params['d1_conn_fn_base']
            fn_out = training_params['d1_merged_conn_fn']
            utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)
      
        print 'Loading BG D1 connections from:', training_params['d1_merged_conn_fn']
        d1_conn_list = np.loadtxt(training_params['d1_merged_conn_fn'])



    def get_weights(self):
        """
        After training get the weights between the MPN state layer and the BG action layer
        """

        print 'Writing weights to files...'
        D1_conns = ''
        D2_conns = ''
        RP_conns = ''   #write code for the RP connections 
        for nactions in range(self.params['n_actions']):
            print 'action %d' % nactions

            conns = nest.GetConnections(self.states, self.strD1[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        D1_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w)

            conns = nest.GetConnections(self.states, self.strD2[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        D2_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w)

        fn_out = self.params['d1_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D1_f = file(fn_out, 'w')
        D1_f.write(D1_conns)
        D1_f.close()

        fn_out = self.params['d2_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D2_f = file(fn_out, 'w')
        D2_f.write(D2_conns)
        D2_f.close()
