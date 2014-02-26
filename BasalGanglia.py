import numpy as np
import nest
import utils
import json


class BasalGanglia(object):

    def __init__(self, params, comm=None):

        self.params = params
        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.comm = comm # mpi communicator needed to broadcast nspikes between processes
        if comm != None:
            assert (comm.rank == self.pc_id), 'mpi4py and NEST tell me different PIDs!'
            assert (comm.size == self.n_proc), 'mpi4py and NEST tell me different PIDs!'

        self.iteration = 0
        self.set_action_speed_mapping_bins() 
        self.strD1 = {}
        self.strD2 = {}
        self.actions = {}
        self.rp = {}
        self.recorder_output= {} # the actual NEST recorder object, indexed by naction
        self.recorder_output_gidkey = {} # here the key is the GID of the spike-recorder and the key is the action --> allows mapping of spike-GID --> action
        self.efference_copy = {}
        self.supervisor = {}
        # Recording devices
        self.recorder_output= {}
        self.recorder_output_gidkey = {}
        self.recorder_d1 = {}
        self.recorder_d2 = {}
        self.recorder_states = {}
        self.recorder_efference = {}
        self.recorder_supervisor = {}
        self.recorder_rp = {}
        self.recorder_rew = nest.Create("spike_detector", params= self.params['spike_detector_rew'])
        nest.SetStatus(self.recorder_rew,[{"to_file": True, "withtime": True, 'label' : self.params['rew_spikes_fn']}])

        #self.recorder_test_rp = nest.Create("spike_detector", params= self.params['spike_detector_test_rp'])
        #nest.SetStatus(self.recorder_test_rp, [{"to_file": True, "withtime": True, 'label' : self.params['test_rp_spikes_fn']}])

        self.voltmeter_d1 = {}
        self.voltmeter_d2 = {}
        self.voltmeter_rp = {}
        self.voltmeter_action = {}
        self.voltmeter_rew = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
        nest.SetStatus(self.voltmeter_rew, [{"to_file": True, "withtime": True, 'label' : self.params['rew_volt_fn']}])

        self.t_current = 0 
#        self.voltmeter_action = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
#        nest.SetStatus(self.voltmeter_action,[{"to_file": True, "withtime": True, 'label' : self.params['bg_action_volt_fn']}])

        #Creates D1 and D2 populations in STRIATUM, connections are created later
        for nactions in range(self.params['n_actions']):
            self.strD1[nactions] = nest.Create(self.params['model_exc_neuron'], self.params['num_msn_d1'], params= self.params['param_msn_d1'])
        for nactions in range(self.params['n_actions']):
            self.strD2[nactions] = nest.Create(self.params['model_inh_neuron'], self.params['num_msn_d2'], params= self.params['param_msn_d2'])

        for nactions in range(self.params['n_actions']):
            self.voltmeter_d1[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
            nest.SetStatus(self.voltmeter_d1[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['strD1_volt_fn']+ str(nactions)}])
            self.voltmeter_d2[nactions] = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.1})
            nest.SetStatus(self.voltmeter_d2[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['strD2_volt_fn']+ str(nactions)}])

        # Creates the different Populations, STR_D1, STR_D2 and Actions, and then create the Connections
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
            nest.SetStatus(self.recorder_d1[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['strD1_spikes_fn']+ str(nactions)}])
            nest.SetStatus(self.recorder_d2[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['strD2_spikes_fn']+ str(nactions)}])
            nest.ConvergentConnect(self.actions[nactions], self.recorder_output[nactions])
            nest.ConvergentConnect(self.strD1[nactions], self.recorder_d1[nactions])
            nest.ConvergentConnect(self.strD2[nactions], self.recorder_d2[nactions])
            for neuron in self.actions[nactions]:
                nest.ConvergentConnect(self.strD1[nactions], [neuron], weight=self.params['str_to_output_exc_w'], delay=self.params['str_to_output_exc_delay']) 
                nest.ConvergentConnect(self.strD2[nactions], [neuron], weight=self.params['str_to_output_inh_w'], delay=self.params['str_to_output_inh_delay'])	

            nest.ConvergentConnect(self.voltmeter_action[nactions], self.actions[nactions])
            nest.RandomConvergentConnect(self.voltmeter_d1[nactions], self.strD1[nactions], int(self.params['random_connect_voltmeter']*self.params['num_msn_d1']))
            nest.RandomConvergentConnect(self.voltmeter_d2[nactions], self.strD2[nactions], int(self.params['random_connect_voltmeter']*self.params['num_msn_d2']))

#            nest.ConvergentConnect(self.voltmeter_action, self.actions[nactions])


        print 'debug recorder gid', self.recorder_output_gidkey
        # create supervisor
        if (self.params['training'] and self.params['supervised_on']):
            print 'DEBUG No supervisor connected'
            for nactions in xrange(self.params['n_actions']):
                self.supervisor[nactions] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_supervisor'], params = self.params['param_poisson_supervisor']  )
                for i in xrange(self.params['n_actions']):
                     if i != nactions:
                         nest.DivergentConnect(self.supervisor[nactions], self.strD2[i], weight=self.params['weight_supervisor_strd2'], delay=self.params['delay_supervisor_strd2'])
                     else:
                         nest.DivergentConnect(self.supervisor[nactions], self.strD1[i], weight=self.params['weight_supervisor_strd1'], delay=self.params['delay_supervisor_strd1'])
                self.recorder_supervisor[nactions] = nest.Create("spike_detector", params= self.params['spike_detector_supervisor'])
                nest.SetStatus(self.recorder_supervisor[nactions],[{"to_file": True, "withtime": True, 'label' : self.params['supervisor_spikes_fn']+ str(nactions)}])
                nest.ConvergentConnect(self.supervisor[nactions], self.recorder_supervisor[nactions])

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


         
        if not self.params['supervised_on']:
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



        ###########################################################
        ############# Now we create bcpnn connections #############

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])
        if not self.params['are_MT_BG_connected']:
            self.create_input_pop()

        if not self.params['supervised_on']:
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

    # used as long as MT and BG are not directly connected
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






    def set_action_speed_mapping_bins(self):
        self.action_bins_x = []
        n_bins_x = np.int(np.round((self.params['n_actions'] - 1) / 2.))
        
        v_scale_half = ((-1.) * np.logspace(np.log(self.params['v_min_tp'])/np.log(self.params['log_scale']),
                            np.log(self.params['v_max_tp'])/np.log(self.params['log_scale']), num=n_bins_x,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        v_scale_half.reverse()

        self.action_bins_x += v_scale_half
        self.action_bins_x += [0.]

        v_scale_half = (np.logspace(np.log(self.params['v_min_tp'])/np.log(self.params['log_scale']),
                            np.log(self.params['v_max_tp'])/np.log(self.params['log_scale']), num=n_bins_x,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        self.action_bins_x += v_scale_half
        print 'BG: action_bins_x', self.action_bins_x


        ### the same for the y-direction
        self.action_bins_y = []
        n_bins_y = np.int(np.round((self.params['n_actions'] - 1) / 2.))
        
        v_scale_half = ((-1.) * np.logspace(np.log(self.params['v_min_tp'])/np.log(self.params['log_scale']),
                            np.log(self.params['v_max_tp'])/np.log(self.params['log_scale']), num=n_bins_y,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        v_scale_half.reverse()

        self.action_bins_y += v_scale_half
        self.action_bins_y += [0.]

        v_scale_half = (np.logspace(np.log(self.params['v_min_tp'])/np.log(self.params['log_scale']),
                            np.log(self.params['v_max_tp'])/np.log(self.params['log_scale']), num=n_bins_y,
                            endpoint=True, base=self.params['log_scale'])).tolist()
        self.action_bins_y += v_scale_half
        print 'BG: action_bins_y', self.action_bins_y

        output_array = np.zeros((len(self.action_bins_x), 2))
#        header = '# first row: action_x, 2nd row: action_y'
        output_array[:, 0] = self.action_bins_x
        output_array[:, 1] = self.action_bins_y
        np.savetxt(self.params['bg_action_bins_fn'], output_array)#, header=header)


    def map_speed_to_action(self, speed, binning, xy='x'):
        # select an action based on the supervisor state information
        if speed > self.params['v_max_tp']:
            action_index = self.params['n_actions'] - 1
        elif (speed < (-1.) * self.params['v_max_tp']):
            action_index = 0
        else:
            cnt_u, bins = np.histogram(speed, binning)
            action_index = cnt_u.nonzero()[0][0]

        if xy == 'x':
            print 'BG.map_speed_to_action (pc_id=%d, iteration=%d) : speed=%.3f --> action: %d ' % (self.pc_id, self.iteration, speed, action_index)
        return action_index


    def supervised_training(self, supervisor_state):
        """
        Activates poisson generator of the required, teached, action and inactivates the one of the nondesirable actions.
        """
        (u, v) = supervisor_state 
        action = [0, 0]
#        action[0] = (x - .5) + u * self.params['t_iteration'] / self.params['t_cross_visual_field']
#        action[1] = (y - .5) + v * self.params['t_iteration'] / self.params['t_cross_visual_field']
        action[0] = u 
        action[1] = v 

#        print 'debug supervisor_state', supervisor_state
#        print 'debug supervisor action', action
        action_index_x = self.map_speed_to_action(action[0], self.action_bins_x, xy='x') # would be interesting to test differences in x/y sensitivity here (as reported from Psychophysics)
        action_index_y = self.map_speed_to_action(action[1], self.action_bins_y, xy='y')

        print 'debug BG based on supervisor action choose action_index_x: %d ~ v_eye = %.2f ' % (action_index_x, self.action_bins_x[action_index_x])
#        action_bins_y = np.linspace(-self.params['v_max_tp'], self.params['v_max_tp'], self.params['n_actions'])
#        cnt_v, bins = np.histogram(action[1], action_bins_y)
#        action_index_y = cnt_v.nonzero()[0][0]

        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})
        nest.SetStatus(self.supervisor[action_index_x], {'rate' : self.params['active_supervisor_rate']})
        # TODO:  same for action_index_y
        
    def stop_efference(self):
        for nactions in xrange(self.params['n_actions']):
            nest.SetStatus(self.efference_copy[nactions], {'rate' : self.params['inactive_efference_rate']})
            print 'debug: EFFERENCE OFF, activity is: ',nest.GetStatus(self.efference_copy[nactions])[0]['rate'] 




    def get_action(self, state):
        """
        Returns the selected action. Calls a selection function e.g. softmax, hardmax, ...
        state -- is (x, y, v_x, v_y) vector-averaging based on MPN spiking activity
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
            gids_spiked = new_event_gids.unique() - 1 # maybe here should be a - 1 (if there is one in communicate_local_spikes)
            nspikes = np.zeros(len(new_event_gids))
            for i_, gid in enumerate(new_event_gids):
                nspikes[i_] = (new_event_gids == gid).nonzero()[0].size
        if len(nspikes) == 0:
            self.t_current += self.params['t_iteration']
            return (0, 0)
        winning_nspikes = np.argmax(nspikes)
        winning_gid = gids_spiked[winning_nspikes]
        print 'winning_gid', winning_gid
        winning_action = self.recorder_output_gidkey[winning_gid+1]
        output_speed_x = self.action_bins_x[winning_action]
        print 'BG says (it %d, pc_id %d): do action %d, output_speed:' % (self.t_current / self.params['t_iteration'], self.pc_id, winning_action), output_speed_x
        self.t_current += self.params['t_iteration']

        self.iteration += 1
        return (output_speed_x, 0)


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

    def get_cell_gids(self, network='strD1'):
        cell_gids = []
        if network == 'strD1':
            for nactions in range(self.params['n_actions']):
                cell_gids.append(self.strD1[nactions])
        elif network == 'strD2':
            for nactions in range(self.params['n_actions']):
                cell_gids.append(self.strD2[nactions])
        elif network == 'actions':
            for nactions in range(self.params['n_actions']):
                cell_gids.append(self.actions[nactions])
        elif network == 'actions':
            for nactions in range(self.params['n_actions']):
                cell_gids.append(self.recorder_output[nactions])
        return cell_gids


    def write_cell_gids_to_file(self):
        d = {}
        cell_types = ['strD1', 'strD2', 'actions', 'recorder']
        for cell_type in cell_types:
            d[cell_type] = self.get_cell_gids(cell_type)
        output_fn = self.params['bg_gids_fn']
        print 'Writing cell_gids to:', output_fn
        f = file(output_fn, 'w')
        json.dump(d, f, indent=2)


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
