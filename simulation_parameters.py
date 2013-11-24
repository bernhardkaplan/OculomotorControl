import os
import numpy as np
import json
import ParameterContainer

class global_parameters(ParameterContainer.ParameterContainer):
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
            self.set_visual_input_params()
            self.set_mpn_params()
            self.set_bg_params()
        else:
            self.params = self.load_params_from_file(params_fn)

        super(global_parameters, self).__init__() # call the constructor of the super/mother class


    def set_default_params(self):
        """
        Here all the simulation parameters NOT being filenames are set.
        """

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['n_training_stim'] = 10 # number of different stimuli presented during training
        self.params['n_testing_stim'] = 10  # number of different stimuli presented during training
        self.params['t_iteration'] = 30.             # [ms] stimulus integration time, after this time the input stimulus will be transformed
        self.params['n_iterations_per_stim'] = 6
        self.params['t_sim'] = (self.params['n_iterations_per_stim']) * self.params['t_iteration'] * self.params['n_training_stim'] # [ms] total simulation time
#        self.params['training'] = True
        self.params['training'] = False
        if self.params['training']:
            self.params['n_iterations'] = self.params['n_training_stim'] * self.params['n_iterations_per_stim']
        else:
            self.params['n_iterations'] = self.params['n_testing_stim'] * self.params['n_iterations_per_stim']
        self.params['dt'] = 0.1                      # [ms]
        self.params['dt_input_mpn'] = 0.1           # [ms] time step for the inhomogenous Poisson process for input spike train generation


        # #####################################
        # CONNECTING MPN --> BG
        # #####################################
        self.params['w_exc_mpn_bg'] = 10.

        # initial motion parameters
        self.params['initial_state'] = (.3, .5, -.2, .0) # initial motion parameters: (x, y, v_x, v_y) position and direction at start


    def set_visual_input_params(self):
        """
        Here only those parameters should be set, which have nothing to do
        with other modules or objects
        TODO: move the cell numbers etc to simulation parameters, 
        as it affects how connections are set up between the MotionPrediction and the BasalGanglia module
        """

        self.params['visual_stim_seed'] = 1234
        self.params['tuning_prop_seed'] = 0
        self.params['dt_stim'] = 1.     # [ms] temporal resolution with which the stimulus trajectory is computed
        self.params['debug_mpn'] = True
        self.params['t_cross_visual_field'] = 1000. # [ms] time in ms for a stimulus with speed 1.0 to cross the whole visual field


        # ###################
        # HEXGRID PARAMETERS
        # ###################
        self.params['n_grid_dimensions'] = 1     # decide on the spatial layout of the network

        self.params['n_rf'] = 50
        if self.params['n_grid_dimensions'] == 2:
            self.params['n_rf_x'] = np.int(np.sqrt(self.params['n_rf'] * np.sqrt(3)))
            self.params['n_rf_y'] = np.int(np.sqrt(self.params['n_rf'])) 
            # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of n_rfdots?"
            self.params['n_theta'] = 3# resolution in velocity norm and direction
        else:
            self.params['n_rf_x'] = self.params['n_rf']
            self.params['n_rf_y'] = 1
            self.params['n_theta'] = 1 # 2 because it's rightwards or leftwards 



    def set_mpn_params(self):

        # #####################################
        # MOTION PREDICTION NETWORK PARAMETERS 
        # #####################################
        self.params['neuron_model_mpn'] = 'iaf_cond_exp'
        self.params['cell_params_exc_mpn'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
                'E_in': -85.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -70.0, 'V_th': -55.0, \
                'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 1.0, 'tau_syn_in': 5.0}
        self.params['cell_params_inh_mpn'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
                'E_in': -85.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -70.0, 'V_th': -55.0, \
                'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 1.0, 'tau_syn_in': 5.0}
        # input parameters
        self.params['w_input_exc_mpn'] = 100. # [nS]
        self.params['f_max_stim'] = 1000.       # [Hz] Max rate of the inhomogenous Poisson process
        # rough values to be chosed for f_max   w_input_exc_mpn
        # for blur_x, v = 0.1, 0.1      4000    50
        #                  .05  .05     5000    100


        # ##############################
        # EXCITATORY NETWORK PARAMETERS
        # ##############################
        # network properties, size, number of preferred directions
        self.params['n_v'] = 12
        assert (self.params['n_v'] % 2 == 0), 'Please choose even number of speeds for even distribution for left/right speed preference'
        self.params['n_hc'] = self.params['n_rf_x'] * self.params['n_rf_y']
        self.params['n_mc_per_hc'] = self.params['n_v'] * self.params['n_theta']
        self.params['n_mc'] = self.params['n_hc'] * self.params['n_mc_per_hc']
        self.params['n_exc_per_state'] = 1
        self.params['n_exc_per_mc'] = self.params['n_exc_per_state']
        self.params['n_exc_mpn'] = self.params['n_mc'] * self.params['n_exc_per_mc']
        print 'n_hc: %d\tn_mc_per_hc: %d\tn_mc: %d\tn_exc_per_mc: %d' % (self.params['n_hc'], self.params['n_mc_per_hc'], self.params['n_mc'], self.params['n_exc_per_mc'])
        self.params['gids_to_record_mpn'] = None
        self.params['log_scale'] = 2.0 # base of the logarithmic tiling of particle_grid; linear if equal to one
        self.params['sigma_rf_pos'] = .05 # some variability in the position of RFs
        self.params['sigma_rf_speed'] = .20 # some variability in the speed of RFs
        self.params['sigma_rf_direction'] = .25 * 2 * np.pi # some variability in the direction of RFs
        self.params['sigma_rf_orientation'] = .1 * np.pi # some variability in the direction of RFs
        self.params['n_exc_to_record_mpn'] = 20
        self.params['v_max_tp'] = 2.0   # [Hz] maximal velocity in visual space for tuning proprties (for each component), 1. means the whole visual field is traversed within 1 second
        self.params['v_min_tp'] = 0.05  # [a.u.] minimal velocity in visual space for tuning property distribution
        self.params['blur_X'], self.params['blur_V'] = .05, .05
        self.params['blur_theta'] = 1.0
        self.params['visual_field_width'] = 1.
        self.params['visual_field_height'] = 1.

        # ##############################
        # INHIBITOTY NETWORK PARAMETERS
        # ##############################
        self.params['fraction_inh_cells_mpn'] = 0.20 # fraction of inhibitory cells in the network, only approximately!
        self.params['n_theta_inh'] = self.params['n_theta']
        self.params['n_v_inh'] = self.params['n_v']
        self.params['n_rf_inh'] = int(round(self.params['fraction_inh_cells_mpn'] * self.params['n_rf']))
        self.params['n_rf_x_inh'] = np.int(np.sqrt(self.params['n_rf_inh'] * np.sqrt(3)))
        # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of n_rf dots?"
        self.params['n_rf_y_inh'] = 1 # np.int(np.sqrt(self.params['n_rf_inh'])) 
        self.params['n_inh_mpn' ] = self.params['n_rf_x_inh'] * self.params['n_rf_y_inh'] * self.params['n_theta_inh'] * self.params['n_v_inh'] * self.params['n_exc_per_mc']
        self.params['n_cells_mpn'] = self.params['n_exc_mpn'] + self.params['n_inh_mpn']
        print 'n_cells_mpn: %d\tn_exc_mpn: %d\tn_inh_mpn: %d\nn_inh_mpn / n_exc_mpn = %.3f\tn_inh_mpn / n_cells_mpn = %.3f' \
                % (self.params['n_cells_mpn'], self.params['n_exc_mpn'], self.params['n_inh_mpn'], \
                self.params['n_inh_mpn'] / float(self.params['n_exc_mpn']), self.params['n_inh_mpn'] / float(self.params['n_cells_mpn']))

                    
        # ############################################################
        # M P N    EXC - INH    C O N N E C T I V I T Y    PARAMETERS
        # ############################################################
        self.params['p_ee_mpn'] = .0 # so far
        self.params['p_ei_mpn'] = .05 # each inh neuron will receive input from p_ei_mpn * n_exc_mpn neurons
        self.params['p_ie_mpn'] = .15 # each exc neuron will receive input from p_ie_mpn * n_inh_mpn neurons
        self.params['p_ii_mpn'] = .0 # ...

        self.params['w_ee_mpn'] = None # so far ...
        self.params['w_ei_mpn'] = 10.  # [nS]
        self.params['w_ie_mpn'] = -40. # [nS]
        self.params['w_ii_mpn'] = -1.  # [nS]
#        self.params['w_ei_mpn'] = 0.  # [nS]
#        self.params['w_ie_mpn'] = 0. # [nS]
#        self.params['w_ii_mpn'] = 0.  # [nS]

        # number of connections to be received by one cell, to get the total number of connections in the network--> multiply by n_tgt_celltype
        self.params['n_ee_mpn'] = int(round(self.params['p_ee_mpn'] * self.params['n_exc_mpn'])) 
        self.params['n_ei_mpn'] = int(round(self.params['p_ei_mpn'] * self.params['n_exc_mpn'])) 
        self.params['n_ie_mpn'] = int(round(self.params['p_ie_mpn'] * self.params['n_inh_mpn'])) 
        self.params['n_ii_mpn'] = int(round(self.params['p_ii_mpn'] * self.params['n_inh_mpn'])) 

        self.params['delay_ee_mpn'] = 1. # [ms]
        self.params['delay_ei_mpn'] = 1. # [ms]
        self.params['delay_ie_mpn'] = 1. # [ms]
        self.params['delay_ii_mpn'] = 1. # [ms]



    def set_bg_params(self):
        """
        Parameters for Basal Ganglia        
        """

        self.params['n_actions'] = 21
        self.params['n_states'] = 20



        #Connections Actions and States to RP
        self.epsilon = 0.01
        self.tau_i = 10.
        self.tau_j = 10.
        self.tau_e = 100.
        self.tau_p = max(1000., self.params['t_sim'])
        self.gain = 0.
        self.K = 50.
        self.params['fmax'] = 20.

        ## State to StrD1/D2 parameters
        self.params['mpn_bg_delay'] = 1.0
        self.params['mpn_bg_weight_amplification'] = 100.

        ## STR
        self.params['model_exc_neuron'] = 'iaf_cond_alpha_bias'
        self.params['model_inh_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_msn_d1'] = 30
        self.params['num_msn_d2'] = 30
        self.params['param_msn_d1'] = {'fmax':self.params['fmax'], 'tau_j': 10.,'tau_e': 100.,'tau_p':100000., 'epsilon': 0.01, 't_ref': 2.0, 'gain': 0.0}
        self.params['param_msn_d2'] = {'fmax':self.params['fmax'], 'tau_j': 10.,'tau_e': 100.,'tau_p':100000., 'epsilon': 0.01, 't_ref': 2.0, 'gain': 0.0}

        
        ## Output GPi/SNr
        self.params['model_bg_output_neuron'] = 'iaf_cond_alpha'
        self.params['num_actions_output'] = 1
        self.params['param_bg_output'] = {'V_reset': -70.0} # to adapt parms to aif_cond_alpha neuron model
        
        self.params['str_to_output_exc_w'] = 10.
        self.params['str_to_output_inh_w'] = -10.
        self.params['str_to_output_exc_delay'] = 1. 
        self.params['str_to_output_inh_delay'] = 1.
        
        ## RP and REWARD
        self.params['model_rp_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_rp_neurons'] = 15
        self.params['param_rp_neuron'] = {'fmax': self.params['fmax'], 'tau_j': 10.,'tau_e': 100.,'tau_p':100000., 'epsilon': 0.01, 't_ref': 2., 'gain': 0.}

        self.params['model_rew_neuron'] = 'iaf_cond_alpha'
        self.params['num_rew_neurons'] = 20
        self.params['param_rew_neuron'] = {} # to adapt parms to aif_cond_alpha neuron model
        self.params['model_poisson_rew'] = 'poisson_generator'
        self.params['num_poisson_rew'] = 20
        self.params['weight_poisson_rew'] = 10.
        self.params['delay_poisson_rew'] = 1.
        self.params['param_poisson_rew'] = {}# to adapt parms to aif_cond_alpha neuron model

        self.params['weight_rp_rew'] = -5. #inhibition of the dopaminergic neurons in rew by the current reward prediction from rp[current state, selected action]
        self.params['delay_rp_rew'] = 1.


        self.params['actions_rp'] = 'bcpnn_synapse'
        self.params['param_actions_rp'] = {'p_i': 0.01, 'p_j': 0.01, 'p_ij': 0.0001, 'gain': self.gain, 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        self.params['states_rp'] = 'bcpnn_synapse'
        self.params['param_states_rp'] = {'p_i': 0.01, 'p_j': 0.01, 'p_ij': 0.0001, 'gain': self.gain, 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}

        self.params['bcpnn'] = 'bcpnn_synapse'
        self.params['param_bcpnn'] =  {'p_i': 0.01, 'p_j': 0.01, 'p_ij': 0.0001, 'gain': self.gain, 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        # during learning gain == 0. K = 1.0 : --> 'offline' learning
        # after learning: gain == 1. K = .0

        #Connections States Actions
        self.params['synapse_d1_MT_BG'] = 'bcpnn_synapse'
        self.params['params_synapse_d1_MT_BG'] = {'p_i': 0.01, 'p_j': 0.01, 'p_ij': 0.0001, 'gain': self.gain, 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        self.params['synapse_d2_MT_BG'] = 'bcpnn_synapse'
        self.params['params_synapse_d2_MT_BG'] = {'p_i': 0.01, 'p_j': 0.01, 'p_ij': 0.0001, 'gain': self.gain, 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}

        #Connections REW to RP, STRD1 and STRD2
        self.params['weight_rew_strD1'] = 10.
        self.params['weight_rew_strD2'] = 10.
        self.params['delay_rew_strD1'] = 1.
        self.params['delay_rew_strD2'] = 1.

        self.params['weight_rew_rp'] = 10.
        self.params['delay_rew_rp'] = 1.

        #Spike detectors params 	
        self.params['spike_detector_output_action'] = {"withgid":True, "withtime":True} 

        #Supervised Learning
        self.params['supervised_on'] = self.params['training']

        self.params['num_neuron_poisson_supervisor'] = 30
        self.params['num_neuron_poisson_input_BG'] = 50
        self.params['active_supervisor_rate'] = 500.
        self.params['inactive_supervisor_rate'] = 0.
        self.params['active_poisson_input_rate'] = 20.
        self.params['inactive_poisson_input_rate'] = 2.
        
        self.params['param_poisson_pop_input_BG'] = {}
        self.params['param_poisson_supervisor'] = {}

        self.params['weight_supervisor_strd1'] = 70.
        self.params['weight_supervisor_strd2'] = 70.
        self.params['delay_supervisor_strd1'] = 1.
        self.params['delay_supervisor_strd2'] = 1.

        self.params['weight_poisson_input'] = 10.
        self.params['delay_poisson_input'] = 1.

        self.params['are_MT_BG_connected'] = True
        
        self.params['num_neuron_states'] = 20
        self.params['param_states_pop'] = {} 

        # some filenames
        self.params['bg_action_volt_fn'] = 'bg_action_volt_'
        self.params['bg_spikes_fn'] = 'bg_spikes_'
        self.params['bg_spikes_fn_merged'] = 'bg_merged_spikes.dat'
        """
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
            self.params[' '] = 
        """

    def set_filenames(self, folder_name=None):
        """
        This funcion is called if no params_fn is passed 
        """

        self.set_folder_names()
        self.params['mpn_exc_volt_fn'] = 'mpn_exc_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['mpn_exc_spikes_fn'] = 'mpn_exc_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['mpn_exc_spikes_fn_merged'] = 'mpn_exc_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['mpn_inh_volt_fn'] = 'mpn_inh_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['mpn_inh_spikes_fn'] = 'mpn_inh_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['mpn_inh_spikes_fn_merged'] = 'mpn_inh_merged_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['training_sequence_fn'] = self.params['parameters_folder'] + 'training_stimuli_parameters.txt'
        
        # input spike files
        self.params['input_st_fn_mpn'] = self.params['input_folder_mpn'] + 'input_spikes_'
        self.params['input_rate_fn_mpn'] = self.params['input_folder_mpn'] + 'input_rate_'
        self.params['input_nspikes_fn_mpn'] = self.params['input_folder_mpn'] + 'input_nspikes_'
        # tuning properties
        self.params['tuning_prop_exc_fn'] = self.params['parameters_folder'] + 'tuning_prop_exc.txt'
        self.params['tuning_prop_inh_fn'] = self.params['parameters_folder'] + 'tuning_prop_inh.txt'
        self.params['gids_to_record_fn_mp'] = self.params['parameters_folder'] + 'gids_to_record_mpn.txt'
        # storage for actions (BG), network states (MPN) and motion parameters (on Retina)
        self.params['actions_taken_fn'] = self.params['data_folder'] + 'actions_taken.txt'
        self.params['network_states_fn'] = self.params['data_folder'] + 'network_states.txt'
        self.params['motion_params_fn'] = self.params['data_folder'] + 'motion_params.txt'

        # connection filenames
        self.params['mpn_bgd1_conn_fn_base'] = self.params['connections_folder'] + 'mpn_bg_d1_connections'
        self.params['mpn_bgd2_conn_fn_base'] = self.params['connections_folder'] + 'mpn_bg_d2_connections'

        self.params['mpn_bgd1_merged_conn_fn'] = self.params['connections_folder'] + 'mpn_bg_d1_merged_connections.txt'
        self.params['mpn_bgd2_merged_conn_fn'] = self.params['connections_folder'] + 'mpn_bg_d2_merged_connections.txt'


    def set_folder_names(self):
#        super(global_parameters, self).set_default_foldernames(folder_name)
#        folder_name = 'Results_GoodTracking_titeration%d/' % self.params['t_iteration']

        if self.params['training']:
            folder_name = 'Training'
        else:
            folder_name = 'Test'

        folder_name += '_nStim%d_nExcMpn%d_nStates%d_nActions%d_it%d-%d/' % \
                (self.params['n_training_stim'], self.params['n_exc_mpn'], self.params['n_states'], \
                self.params['n_actions'], self.params['t_iteration'], self.params['t_sim'])

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


        self.params['input_folder_mpn'] = '%sInputSpikes_MPN/' % (self.params['folder_name'])
        self.params['spiketimes_folder_mpn'] = '%sSpikes/' % self.params['folder_name']
        self.params['folder_names'].append(self.params['spiketimes_folder_mpn'])
        self.params['folder_names'].append(self.params['input_folder_mpn'])
        self.create_folders()


    def load_params_from_file(self, fn):
        """
        Loads the file via json from a filename
        Returns the simulation parameters stored in a file 
        Keyword arguments:
        fn -- file name
        """
        f = file(fn, 'r')
        print 'Loading parameters from', fn
        self.params = json.load(f)
        return self.params


