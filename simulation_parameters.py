import os
import numpy as np
import json
import ParameterContainer

class global_parameters(ParameterContainer.ParameterContainer):
#class global_parameters(object):
    """
    The parameter class storing the simulation parameters 
    is derived from the ParameterContainer class.

    Parameters used (mostly) by different classes should be seperated into
    different functions.
    Common parameters are set in the set_default_params function
    """

    def __init__(self, params_fn=None):#, output_dir=None):
        """
        Keyword arguments:
        params_fn -- string, if None: set_filenames and set_default_params will be called
        """
        

        if params_fn == None:
            self.params = {}
            self.set_default_params()
            self.set_input_params()
            self.set_bg_params()
        else:
            self.load_params_from_file(params_fn)

        super(global_parameters, self).__init__() # call the constructor of the super/mother class

    def set_default_params(self):
        """
        Here all the simulation parameters NOT being filenames are set.
        """

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_selection'] = 20.
        self.params['t_efference'] = 10
        self.params['t_reward'] = 10.
        self.params['t_rest'] = 10. 
        self.params['t_iteration'] = self.params['t_selection'] + self.params['t_efference'] + self.params['t_rest'] + self.params['t_reward']  # [ms] stimulus integration time, 
                                                                                                               # after this time the input stimulus will be transformed
        self.params['block_len'] = 100
        self.params['n_blocks'] = 1

        self.params['t_sim'] = self.params['t_iteration'] * self.params['block_len'] * self.params['n_blocks']                 # [ms] total simulation time
        
        
        self.params['dt'] = 0.1                      # [ms]
        self.params['n_iterations'] = self.params['block_len'] * self.params['n_blocks']  #int(round(2*self.params['t_sim'] / self.params['t_iteration']))
        self.params['dt_input_mpn'] = 0.1           # [ms] time step for the inhomogenous Poisson process for input spike train generation


    def set_input_params(self):
	    pass


        # ##############################
        # INPUT PARAMETERS
        # ##############################

    def set_bg_params(self):
        """
        Parameters for Basal Ganglia        
        """

        self.params['n_actions'] = 5
        self.params['n_states'] = 6

        self.params['random_divconnect_poisson'] = 0.75
        self.params['random_connect_voltmeter'] = 0.05
        
        
        self.epsilon = 0.0001
        self.tau_i = 5.
        self.tau_j = 5.
        self.tau_e = 20.
        self.tau_p = 1000.
        self.gain = 0.
        self.K = 1.
        self.fmax = 50.
        
        ## STR
        self.params['model_exc_neuron'] = 'iaf_cond_alpha_bias'
        self.params['model_inh_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_msn_d1'] = 60
        self.params['num_msn_d2'] = 60
        self.params['param_msn_d1'] = {'kappa': self.K ,'fmax':self.fmax, 'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p':self.tau_p, 'epsilon': self.epsilon, 't_ref': 2.0, 'gain': self.gain}
        self.params['param_msn_d2'] = {'kappa': self.K ,'fmax':self.fmax, 'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p':self.tau_p, 'epsilon': self.epsilon, 't_ref': 2.0, 'gain': self.gain}

        
        ## Output GPi/SNr
        self.params['model_bg_output_neuron'] = 'iaf_cond_alpha'
        self.params['num_actions_output'] = 10
        self.params['param_bg_output'] = {'V_reset': -70.0} # to adapt parms to aif_cond_alpha neuron model
        
        self.params['str_to_output_exc_w'] = 50.
        self.params['str_to_output_inh_w'] = -50.
        self.params['str_to_output_exc_delay'] = 1. 
        self.params['str_to_output_inh_delay'] = 1.
        
        ## RP and REWARD
        self.params['model_rp_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_rp_neurons'] = 15
        self.params['param_rp_neuron'] = {'kappa': self.K ,'fmax':self.fmax, 'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p':self.tau_p, 'epsilon': self.epsilon, 't_ref': 2.0, 'gain': self.gain}


        self.params['model_rew_neuron'] = 'iaf_cond_alpha'
        self.params['num_rew_neurons'] = 50
        self.params['param_rew_neuron'] = {} # to adapt parms to aif_cond_alpha neuron model
        self.params['model_poisson_rew'] = 'poisson_generator'
        self.params['num_poisson_rew'] = 30
        self.params['weight_poisson_rew'] = 10.
        self.params['delay_poisson_rew'] = 1.
        self.params['param_poisson_rew'] = {}# to adapt parms to aif_cond_alpha neuron model

        self.params['weight_rp_rew'] = -5. #inhibition of the dopaminergic neurons in rew by the current reward prediction from rp[current state, selected action]
        self.params['delay_rp_rew'] = 1.


        #Connections Actions and States to RP
        
        self.p_i = 0.01
        self.p_j = 0.01
        self.p_ij= 0.0002

        self.params['actions_rp'] = 'bcpnn_synapse'
        self.params['param_actions_rp'] = {'p_i': self.p_i, 'p_j': self.p_j, 'p_ij': self.p_ij, 'gain': self.gain, 'K': self.K,'fmax': self.fmax,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        self.params['states_rp'] = 'bcpnn_synapse'
        self.params['param_states_rp'] = {'p_i': self.p_i, 'p_j': self.p_j, 'p_ij': self.p_ij, 'gain': self.gain, 'K': self.K,'fmax': self.fmax,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}

        self.params['bcpnn'] = 'bcpnn_synapse'
        self.params['param_bcpnn'] = {'p_i': self.p_i, 'p_j': self.p_j, 'p_ij': self.p_ij, 'gain': self.gain, 'K': self.K,'fmax': self.fmax,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p} 
        # during learning gain == 0. K = 1.0 : --> 'offline' learning
        # after learning: gain == 1. K = .0

        #Connections States Actions
        self.params['synapse_d1'] = 'bcpnn_synapse'
        self.params['params_synapse_d1'] = {'p_i': self.p_i, 'p_j': self.p_j, 'p_ij': self.p_ij, 'gain': self.gain, 'K': self.K,'fmax': self.fmax,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p} 
        self.params['synapse_d2'] = 'bcpnn_synapse'
        self.params['params_synapse_d2'] = {'p_i': self.p_i, 'p_j': self.p_j, 'p_ij': self.p_ij, 'gain': self.gain, 'K': self.K,'fmax': self.fmax,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p} 

        #Connections REW to RP, STRD1 and STRD2
        self.params['weight_rew_strD1'] = 10.
        self.params['weight_rew_strD2'] = 10.
        self.params['delay_rew_strD1'] = 1.
        self.params['delay_rew_strD2'] = 1.

        self.params['weight_rew_rp'] = 30.
        self.params['delay_rew_rp'] = 1.

        #Spike detectors params 	
        self.params['spike_detector_action'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_d1'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_d2'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_states'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_efference'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_rew'] = {"withgid":True, "withtime":True}
        self.params['spike_detector_rp'] = {"withgid":True, "withtime":True}

        self.params['spike_detector_test_rp'] = {"withgid":True, "withtime":True}
 

	    #Reinforcement Learning


        self.params['num_neuron_poisson_efference'] = 40
        self.params['num_neuron_poisson_input_BG'] = 100
        self.params['active_full_efference_rate'] = 40.
        self.params['inactive_efference_rate'] = 0.
        self.params['active_poisson_input_rate'] = 50.
        self.params['inactive_poisson_input_rate'] = 0.5
        self.params['supervisor_off'] = 0.

        self.params['active_poisson_rew_rate'] = 70.
        self.params['inactive_poisson_rew_rate'] = 1.
        
        self.params['param_poisson_pop_input_BG'] = {}
        self.params['param_poisson_efference'] = {}

        self.params['weight_efference_strd1'] = 15.
        self.params['weight_efference_strd2'] = 15.
        self.params['delay_efference_strd1'] = 1.
        self.params['delay_efference_strd2'] = 1.

        self.params['weight_poisson_input'] = 20.
        self.params['delay_poisson_input'] = 1.
        
        self.params['num_neuron_states'] = 30
        self.params['param_states_pop'] = {} 


        """
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
        """


        # ##############################
        # RECORDING PARAMETERS
        # ##############################

    def set_recorders(self):

	 pass

        

    def set_filenames(self, folder_name=None):
        """
        This function is called if no params_fn is passed 
        """

        self.set_folder_names()
        
        self.params['states_spikes_fn'] = 'states_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_spikes_fn'] = 'd1_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_volt_fn'] = 'd1_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_spikes_fn'] = 'd2_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_volt_fn'] = 'd2_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['actions_spikes_fn'] = 'actions_spikes_'
        self.params['actions_volt_fn'] = 'actions_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['efference_spikes_fn'] = 'efference_spikes_'
        self.params['rew_spikes_fn'] = 'rew_spikes_'
        self.params['rew_volt_fn'] = 'rew_volt_'
        self.params['rp_spikes_fn'] = 'rp_spikes_'
        self.params['rp_volt_fn'] = 'rp_volt_'
        
        self.params['test_rp_spikes_fn'] = 'test_rp_spikes_'
        
        self.params['states_spikes_fn_merged'] = 'states_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_spikes_fn_merged'] = 'd1_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d1_volt_fn_merged'] = 'd1_merged_volt.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_volt_fn_merged'] = 'd2_merged_volt.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['d2_spikes_fn_merged'] = 'd2_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['actions_spikes_fn_merged'] = 'actions_merged_spikes.dat'
        self.params['actions_volt_fn_merged'] = 'actions_merged_volt.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['efference_spikes_fn_merged'] = 'efference_merged_spikes.dat'
        self.params['rew_spikes_fn_merged'] = 'rew_merged_spikes.dat'
        self.params['rew_volt_fn_merged'] = 'rew_merged_volt.dat'
        self.params['rp_spikes_fn_merged'] = 'rp_merged_spikes.dat'
        self.params['rp_volt_fn_merged'] = 'rp_merged_volt.dat'

        # input spike files
#        self.params['input_st_fn_mpn'] = self.params['input_folder_mpn'] + 'input_spikes_'
#        self.params['input_rate_fn_mpn'] = self.params['input_folder_mpn'] + 'input_rate_'
#        self.params['input_nspikes_fn_mpn'] = self.params['input_folder_mpn'] + 'input_nspikes_'
        # tuning properties
#        self.params['tuning_prop_exc_fn'] = self.params['parameters_folder'] + 'tuning_prop_exc.txt'
#        self.params['tuning_prop_inh_fn'] = self.params['parameters_folder'] + 'tuning_prop_inh.txt'
#        self.params['gids_to_record_fn_mp'] = self.params['parameters_folder'] + 'gids_to_record_mpn.txt'
        # storage for actions (BG), network states (MPN) and motion parameters (on Retina)
        self.params['actions_taken_fn'] = self.params['data_folder'] + 'actions_taken.txt'
        self.params['states_fn'] = self.params['data_folder'] + 'states.txt'
        self.params['rewards_fn'] = self.params['data_folder'] + 'rewards.txt'
#        self.params['motion_params_fn'] = self.params['data_folder'] + 'motion_params.txt'

        # connection filenames
        self.params['d1_conn_fn_base'] = self.params['connections_folder'] + 'd1_connections'
        self.params['d2_conn_fn_base'] = self.params['connections_folder'] + 'd2_connections'

        self.params['d1_weights_fn'] = self.params['connections_folder'] + 'd1_merged_connections.txt'
        self.params['d2_weights_fn'] = self.params['connections_folder'] + 'd2_merged_connections.txt'


    def set_folder_names(self):
    #    super(global_parameters, self).set_default_foldernames(folder_name)
    #    folder_name = 'Results_GoodTracking_titeration%d/' % self.params['t_iteration']
        folder_name = 'Test/'

#        if self.params['supervised_on'] == True:
#            folder_name += '_WithSupervisor/'
#        else:
#            folder_name += '_NoSupervisor/'
        assert(folder_name[-1] == '/'), 'ERROR: folder_name must end with a / '

        self.set_folder_name(folder_name)

        self.params['parameters_folder'] = "%sParameters/" % self.params['folder_name']
        self.params['figures_folder'] = "%sFigures/" % self.params['folder_name']
        self.params['connections_folder'] = "%sConnections/" % self.params['folder_name']
        self.params['tmp_folder'] = "%stmp/" % self.params['folder_name']
        self.params['data_folder'] = '%sData/' % (self.params['folder_name']) # for storage of analysis results etc
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['parameters_folder'], \
                            self.params['figures_folder'], \
                            self.params['tmp_folder'], \
                            self.params['connections_folder'], \
                            self.params['data_folder']]

        self.params['params_fn_json'] = '%ssimulation_parameters.json' % (self.params['parameters_folder'])


#        self.params['input_folder_mpn'] = '%sInputSpikes_MPN/' % (self.params['folder_name'])
        self.params['spiketimes_folder'] = '%sSpikes/' % self.params['folder_name']
        self.params['folder_names'].append(self.params['spiketimes_folder'])
#        self.params['folder_names'].append(self.params['input_folder_mp'])
        self.create_folders()

