import numpy as np
import nest



class BasalGanglia(object):

    def __init__(self, params):
        
        self.params = params
	self.str_d1 = {}
	self.str_d2 = {}
	self.actions = {}

# create the different Populations, STR_D1, STR_D2 and Actions, and then create the Connections
	for nactions in range(0, self.params['n_actions']):
	    self.str_d1[nactions] = nest.Create(self.params['model_exc_neuron'], self.params['num_msn_d1'], params= self.params['param_msn_d1'])
            self.str_d2[nactions] = nest.Create(self.params['model_inh_neuron'], self.params['num_msn_d2'], params= self.params['param_msn_d2'])
	    self.actions[nactions] = nest.Create(self.params['model_bg_output_neuron'], self.params['num_actions_output'], params= self.params['param_bg_output'])
	    nest.ConvergentConnect(self.str_d1[nactions], self.actions[nactions], weight=self.params['str_to_output_exc_w'], delay=self.params['str_to_output_exc_delay']) 
	    nest.ConvergentConnect(self.str_d2[nactions], self.actions[nactions], weight=self.params['str_to_output_inh_w'], delay=self.params['str_to_output_inh_delay']) 	


	if self.params['supervised_on']:
	# used to create external poisson process to make the system learns the desired output action by stimulating its relative striatal D1 population and the striatal D2 populations associated with the other actions.
	    self.supervisor = {}
	    for nactions in range(0, self.params['n_actions']):
		self.supervisor[nactions] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_supervisor'], params = self.params['param_poisson_supervisor']  )
		for supervisor_neuron in self.supervisor[nactions]:
		    for i in range(0, self.params['n_actions']):
			if i != nactions:
			    nest.DivergentConnect([self.supervisor_neuron], self.str_d2[i], weight=self.params['weight_supervisor_strd2'], delay=self.params['delay_supervisor_strd2'])
		    nest.DivergentConnect( [self.supervisor_neuron], self.str_d1[nactions], weight=self.params['weight_supervisor_strd1'], delay=self.params['delay_supervisor_strd1'] )



	if not self.params['are_MT_BG_connected']:
	    self.create_input_pop()

	print "BG model completed"

	pass

# used as long as MT and BG are not directly connected
    def create_input_pop(self):
	"""
	Creates the input states populations, and their respective poisson pop,  and connect them to Striatum msns D1 D2 populations
	"""
	self.states = {}
	self.input_poisson = {}

	for nstates in range(0, self.params['n_states']):
	    self.input_poisson[nstates] = nest.Create( 'poisson_generator', self.params['num_neuron_poisson_input_BG'], params = self.params['param_poisson_pop_input_BG']  )
	    self.states[nstates] = nest.Create( self.params['model_exc_neuron'], self.params['num_neuron_states'], params = self.params['param_states_pop']  )
	    for neuron_poisson in self.input_poisson[nstates]:
		nest.DivergentConnect([self.neuron_poisson], self.states[nstates], weight=self.params['weight_poisson_input'], delay=self.params['delay_poisson_input'])
 	    for neuron_input in self.states[nstates]:
		for nactions in range(0, self.params['n_actions']):
		    nest.DivergentConnect([neuron_input], self.str_d1[nactions], model=self.params['synapse_d1_MT_BG'], params = self.params['params_synapse_d1_MT_BG'])
		    nest.DivergentConnect([neuron_input], self.str_d2[nactions], model=self.params['synapse_d2_MT_BG'], params = self.params['params_synapse_d2_MT_BG'])
	
	print "BG input stage created"

	pass


    def supervised_training(self, action_index):
        """
	Activates poisson generator of the required, teached, action and inactivates the one of the nondesirable actions.
	"""
	for nactions in range(0, self.params['n_actions']):
	    nest.SetStatus(self.supervisor[nactions], {'rate' : self.params['inactive_supervisor_rate']})
	nest.SetStatus(self.supervisor[action_index], {'rate' : self.params['active_supervisor_rate']})
	pass


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
