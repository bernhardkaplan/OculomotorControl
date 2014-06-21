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

        # will set the filenames
        super(global_parameters, self).__init__() # call the constructor of the super/mother class


    def set_default_params(self):
        """
        Here all the simulation parameters NOT being filenames are set.
        """

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['Cluster'] = False
        self.params['Cluster_Milner'] = False
        self.params['total_num_virtual_procs'] = 8
        if self.params['Cluster'] or self.params['Cluster_Milner']:
            self.params['total_num_virtual_procs'] = 160
        self.params['n_rf'] = 50
        self.params['n_v'] = 40
#        self.params['n_rf'] = 40
        self.params['n_v'] = 30

#        self.params['training'] = False
        self.params['training'] = True
        self.params['reward_based_learning'] = True

        self.params['n_training_cycles'] = 1            # how often each stimulus is presented during training
#        if self.params['reward_based_learning']:
#            assert (self.params['n_training_cycles'] % 2) == 0, 'Each stimulus needs to be presented twice (once with plasiticity off to get the reward signal, \
#                    once with plasticity on when reward signal has arrived and the efference copy re-activating stimulus and D1/D2 activity'
        self.params['n_training_x'] = 2 # number of training samples to cover the x-direction of the tuning space
        self.params['n_training_v'] = 1 # number of training samples to cover the v-direction of the tuning space
        self.params['n_training_stim_per_cycle'] = self.params['n_training_x'] * self.params['n_training_v']
        self.params['n_stim_training'] = self.params['n_training_cycles'] * self.params['n_training_stim_per_cycle'] # total number of stimuli presented during training

#        self.params['train_iteratively'] = False
        self.params['test_stim_range'] = range(0, 3)
        if len(self.params['test_stim_range']) > 1:
            self.params['n_stim_testing'] = len(self.params['test_stim_range'])
        else:
            self.params['n_stim_testing'] = 1
        self.params['t_iteration'] = 25.   # [ms] stimulus integration time, after this time the input stimulus will be transformed
        self.params['n_silent_iterations'] = 2 # for 2 silent iterations this should be 3
        if self.params['training']:
            if self.params['reward_based_learning']:
                self.params['n_iterations_per_stim'] = (2 + self.params['n_silent_iterations'])
            else:
                # 'open-loop': 
                self.params['n_iterations_per_stim'] = 1 + self.params['n_silent_iterations'] 

            # else:
        else:
            self.params['n_iterations_per_stim'] = 20 + self.params['n_silent_iterations']
        # effective number of training iterations is n_iterations_per_stim - n_silent_iterations
        self.params['weight_tracking'] = False # if True weights will be written to file after each iteration --> use only for debugging / plotting
        # if != 0. then weights with abs(w) < 
        self.params['connect_d1_after_training'] = True
        self.params['clip_weights_mpn_d1'] = False # only for VisualLayer --> D1 weights
        self.params['clip_weights_mpn_d2'] = self.params['clip_weights_mpn_d1']
        self.params['clip_weights_d1_d1'] = True # only for VisualLayer --> D1 weights
        self.params['weight_threshold_abstract_mpn_d1'] = (.05, False)  # (value for thresholding, absolute_values considered for thresholding)
        # if (THRESH, True) --> (abs(x)>thresh, -x) after thresholding abstract weights can contain also negative weights abs(w) > THRESH
        # if (THRESH, False) --> (x > thresh, x) x > 0 after thresholding abstract weights are all positive and > THRESH
        self.params['weight_threshold_abstract_mpn_d2'] = self.params['weight_threshold_abstract_mpn_d1']
        self.params['weight_threshold_abstract_d1_d1'] = (.05, True)  # (value for thresholding, absolute_values considered for thresholding)

        # For reward-based learning:
#        self.params['load_mpn_d1_weights'] = False
#        self.params['load_mpn_d2_weights'] = False

        if self.params['training']:
            self.params['n_stim'] = self.params['n_stim_training']
        else:
            self.params['n_stim'] = self.params['n_stim_testing']
        self.params['n_iterations'] = (self.params['n_iterations_per_stim']) * self.params['n_stim'] # [ms] total simulation time 
        self.params['t_sim'] = (self.params['n_iterations_per_stim']) * self.params['t_iteration'] * self.params['n_stim'] # [ms] total simulation time 
        self.params['dt'] = 0.1            # [ms] simulation time step
        self.params['dt_input_mpn'] = 0.1  # [ms] time step for the inhomogenous Poisson process for input spike train generation
        self.params['dt_volt'] = 0.1       # [ms] time step for volt / multimeter

        # the first stimulus parameters
        self.params['initial_state'] = (.8, .5, -2.0, .0)
#        self.params['initial_state'] = (.631059, .5, 0.1996527, .0)
        assert (self.params['n_v'] % 2 == 0), 'Please choose even number of speeds for even distribution for left/right speed preference'
        self.params['v_min_out'] = 0.1  # min velocity for eye movements
        self.params['v_max_out'] = 12.0   # max velocity for eye movements (for humans ~900 degree/sec, i.e. if screen for stimulus representation (=visual field) is 45 debgree of the whole visual field (=180 degree))
        self.params['t_iter_training'] = 50
        if self.params['training']:
            self.params['sim_id'] = 'titer%d_nRF%d_nV%d_vmin%.2f_vmax%.2f' % (self.params['t_iteration'], self.params['n_rf'], self.params['n_v'], self.params['v_min_out'], self.params['v_max_out'])
            if (self.params['reward_based_learning']):
                self.params['sim_id'] = 'RBL_titer%d_nRF%d_nV%d_vmin%.2f_vmax%.2f' % (self.params['t_iteration'], self.params['n_rf'], self.params['n_v'], self.params['v_min_out'], self.params['v_max_out'])
        else:
            self.params['sim_id'] = 'titer_train%d_test%d_vmin%.2f_vmax%.2f' % (self.params['t_iter_training'], self.params['t_iteration'], self.params['v_min_out'], self.params['v_max_out'])

#        self.params['initial_state'] = (.3, .5, -.2, .0) # initial motion parameters: (x, y, v_x, v_y) position and direction at start


    def set_visual_input_params(self):
        """
        Here only those parameters should be set, which have nothing to do
        with other modules or objects
        TODO: move the cell numbers etc to simulation parameters, 
        as it affects how connections are set up between the MotionPrediction and the BasalGanglia module
        """

        self.params['master_seed'] = 12345  # 
        np.random.seed(self.params['master_seed'])
        # one global seed for calculating the tuning properties and the visual stim properties (not the spiketrains)
        self.params['visual_stim_seed'] = 1234
        self.params['tuning_prop_seed'] = 0
        self.params['dt_stim'] = 1.     # [ms] temporal resolution with which the stimulus trajectory is computed
#        self.params['debug_mpn'] = False
        self.params['debug_mpn'] = not self.params['Cluster']
        self.params['t_cross_visual_field'] = 1000. # [ms] time in ms for a stimulus with speed 1.0 to cross the whole visual field


        # ###################
        # HEXGRID PARAMETERS
        # ###################
        self.params['n_grid_dimensions'] = 1     # decide on the spatial layout of the network

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
#        self.params['neuron_model_mpn'] = 'iaf_cond_exp_bias'
        self.params['neuron_model_mpn'] = 'iaf_cond_exp'
        self.params['cell_params_exc_mpn'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
                'E_in': -80.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -80.0, 'V_th': -50.0, \
                'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 5.0, 'tau_syn_in': 5.0}
        self.params['cell_params_inh_mpn'] = self.params['cell_params_exc_mpn'].copy()

#        self.params['cell_params_inh_mpn'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
#                'E_in': -80.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -70.0, 'V_th': -55.0, \
#                'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 2.0, 'tau_syn_in': 5.0}
        # input parameters
        self.params['w_input_exc_mpn'] = 30. # [nS]
        self.params['f_max_stim'] = 2000.       # [Hz] Max rate of the inhomogenous Poisson process
        # rough values to be chosed for f_max   w_input_exc_mpn
        # for blur_x, v = 0.1, 0.1      4000    50
        #                  .05  .05     5000    100

        self.params['f_noise_exc'] = 1000.
        self.params['f_noise_inh'] = 1000.
        self.params['w_noise_exc'] = .2
        self.params['w_noise_inh'] = -.2

        # ##############################
        # EXCITATORY NETWORK PARAMETERS
        # ##############################
        # network properties, size, number of preferred directions
        self.params['n_hc'] = self.params['n_rf_x'] * self.params['n_rf_y']
        self.params['n_mc_per_hc'] = self.params['n_v'] * self.params['n_theta']
        self.params['n_mc'] = self.params['n_hc'] * self.params['n_mc_per_hc']
        self.params['n_exc_per_state'] = 1
        self.params['n_exc_per_mc'] = self.params['n_exc_per_state']
        self.params['n_exc_mpn'] = self.params['n_mc'] * self.params['n_exc_per_mc']
        print 'n_hc: %d\tn_mc_per_hc: %d\tn_mc: %d\tn_exc_per_mc: %d' % (self.params['n_hc'], self.params['n_mc_per_hc'], self.params['n_mc'], self.params['n_exc_per_mc'])
        # most active neurons for certain iterations can be determined by PlottingScripts/plot_bcpnn_traces.py
        self.params['gids_to_record_mpn'] = None
        self.params['gids_to_record_bg'] = None

#        self.params['gids_to_record_mpn'] = [270, 365, 502, 822, 1102, 1108, 1132, 1173, 1174, 1437, 1510, 1758, 1797, 2277, 2374, 2589, 2644, 3814, 4437, 4734, 4821, 4989, 5068, 5134, 5718, 6021, 6052, 6318, 7222, 7246, 7396, 7678, 8014, 8454, 8710, 8973, 9052, 9268, 9438, 9669, 10014, 10247, 10398, 10414, 10492, 11214, 11349, 11637]
#        self.params['gids_to_record_bg'] = [57006, 57007, 57011, 57013, 57030, 57032, 57033, 57034, 57035, 57036, 57037, 57038, 57041, 57042, 57043, 57089, 57090, 57091, 57092, 57093, 57096, 57097, 57098, 57102, 57103, 57107, 57108]

        self.params['log_scale'] = 2.0 # base of the logarithmic tiling of particle_grid; linear if equal to one
        self.params['sigma_rf_pos'] = .25 # RF are drawn from a normal distribution centered at 0.5 with this sigma as standard deviation
        self.params['sigma_rf_speed'] = .20 # some variability in the speed of RFs
        self.params['sigma_rf_direction'] = .25 * 2 * np.pi # some variability in the direction of RFs
        self.params['sigma_rf_orientation'] = .1 * np.pi # some variability in the direction of RFs
        self.params['n_exc_to_record_mpn'] = 0
        self.params['x_max_tp'] = 1.0   # [a.u.] maximal position in visual space for tuning properties
        self.params['x_min_tp'] = 0.0   # [a.u.] minimal position in visual space for tuning properies
        self.params['v_max_tp'] = 2.0   # [a.u.] maximal velocity in visual space for tuning properties (for each component), 1. means the whole visual field is traversed within 1 second
        self.params['v_min_tp'] = 0.01  # [a.u.] minimal velocity in visual space for tuning property distribution
#        self.params['v_max_out'] = 12.0   # max velocity for eye movements (for humans ~900 degree/sec, i.e. if screen for stimulus representation (=visual field) is 45 debgree of the whole visual field (=180 degree))
        self.params['blur_X'], self.params['blur_V'] = .2, .3
        self.params['training_stim_noise_x'] = 0.05 # noise to be applied to the training stimulus parameters (absolute, not relative to the 'pure stimulus parameters')
        self.params['training_stim_noise_v'] = 0.10 # noise to be applied to the training stimulus parameters (absolute, not relative to the 'pure stimulus parameters')
        self.params['blur_theta'] = 1.0

        self.params['visual_field_width'] = 1.
        self.params['visual_field_height'] = 1.
        # receptive field size parameters
        # receptive field sizes are determined by their relative position (for x/y relative to .5, for u/v relative to 0.)
        # rf_size = rf_size_gradient * |relative_rf_pos| + min_rf_size
        # check for reference: Dow 1981 "Magnification Factor and Receptive Field Size in Foveal Striate Cortex of the Monkey"
        self.params['regular_tuning_prop'] = False
        if self.params['regular_tuning_prop']:
            # regular tuning prop
            self.params['rf_size_x_gradient'] = .0  # receptive field size for x-pos increases with distance to .5
            self.params['rf_size_y_gradient'] = .0  # receptive field size for y-pos increases with distance to .5
            self.params['rf_size_x_min'] = (self.params['x_max_tp'] - self.params['x_min_tp']) / self.params['n_rf_x']
            self.params['rf_size_y_min'] = (self.params['x_max_tp'] - self.params['x_min_tp']) / self.params['n_rf_y']
            self.params['rf_size_vx_gradient'] = .0 # receptive field size for vx-pos increases with distance to 0.0
            self.params['rf_size_vy_gradient'] = .0 #
            self.params['rf_size_vx_min'] = (self.params['v_max_tp'] - self.params['v_min_tp']) / self.params['n_v']
            self.params['rf_size_vy_min'] = (self.params['v_max_tp'] - self.params['v_min_tp']) / self.params['n_v']
        else:
            self.params['rf_size_x_gradient'] = .2  # receptive field size for x-pos increases with distance to .5
            self.params['rf_size_y_gradient'] = .2  # receptive field size for y-pos increases with distance to .5
#            self.params['rf_size_x_min'] = .01      # cells situated at .5 have this receptive field size
#            self.params['rf_size_y_min'] = .01      # cells situated at .5 have this receptive field size
            self.params['rf_size_x_min'] = (self.params['x_max_tp'] - self.params['x_min_tp']) / self.params['n_rf_x']
            self.params['rf_size_y_min'] = (self.params['x_max_tp'] - self.params['x_min_tp']) / self.params['n_rf_y']
            self.params['rf_size_vx_gradient'] = .3 # receptive field size for vx-pos increases with distance to 0.0
            self.params['rf_size_vy_gradient'] = .3 #
#            self.params['rf_size_vx_min'] = .01 # cells situated at .5 have this receptive field size
#            self.params['rf_size_vy_min'] = .01 # cells situated at .5 have this receptive field size
            self.params['rf_size_vx_min'] = (self.params['v_max_tp'] - self.params['v_min_tp']) / self.params['n_v']
            self.params['rf_size_vy_min'] = (self.params['v_max_tp'] - self.params['v_min_tp']) / self.params['n_v']


        # during training a supervisor signal is generated based on displacement and perceived speed, using this parameter
        self.params['supervisor_amp_param'] = 1.

        # ##############################
        # INHIBITOTY NETWORK PARAMETERS
        # ##############################
        self.params['fraction_inh_cells_mpn'] = 0.25 # fraction of inhibitory cells in the network, only approximately!
        self.params['n_theta_inh'] = self.params['n_theta']
        self.params['n_v_inh'] = self.params['n_v']
        self.params['n_rf_inh'] = int(round(self.params['fraction_inh_cells_mpn'] * self.params['n_rf']))
        self.params['n_rf_x_inh'] = self.params['n_rf_inh']
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

        if self.params['Cluster'] or self.params['Cluster_Milner']:
            self.params['record_bg_volt'] = False
        else:
            self.params['record_bg_volt'] = True
#        self.params['record_bg_volt'] = False
        self.params['bg_cell_types'] = ['d1', 'd2', 'actions', 'recorder']
        self.params['n_actions'] = 17
        self.params['n_states'] = 16
        self.params['random_divconnect_poisson'] = 0.75
        self.params['random_connect_voltmeter'] = 0.20

        #Connections Actions and States to RP
        self.tau_i = 5.
        self.tau_j = 5.
        self.tau_e = 5.
#        self.au_p = max(1000., self.params['t_sim'])
#        self.tau_p = .5 * self.params['t_sim']
        self.tau_p = 1. * self.params['t_sim']
        self.params['fmax'] = 150.
        self.epsilon = 1. / (self.params['fmax'] * self.tau_p)
        if self.params['training']:# and not self.params['reward_based_learning']:
            self.params['gain'] = 0.
            if self.params['reward_based_learning']:
                self.K = 0. # will be set to 1 during the main learning script
            else:
                self.K = 1.
        else:
            self.params['gain'] = 1.
            self.K = 0.
        self.params['kappa'] = self.K


        # #####################################
        # CONNECTING MPN --> BG
        # #####################################
        ## State to StrD1/D2 parameters
        self.params['mpn_bg_delay'] = 1.0
        self.params['weight_threshold'] = 0.05
        self.params['mpn_d1_weight_amplification'] = 2.5
        self.params['mpn_d2_weight_amplification'] = 2.5
        self.params['mpn_bg_bias_amplification'] = 0.1
        self.params['d1_d1_weight_amplification_neg'] = 5.0
        self.params['d1_d1_weight_amplification_pos'] = 0.0
        # if static synapses are used
        self.params['w_d1_d1'] = -5.
        self.params['delay_d1_d1'] = 1.
        self.params['w_d2_d2'] = 0.
        self.params['delay_d2_d2'] = 1.

        ## STR
#        self.params['model_exc_neuron'] = 'iaf_cond_exp'
#        self.params['model_inh_neuron'] = 'iaf_cond_exp'
        self.params['model_exc_neuron'] = 'iaf_cond_exp_bias'
        self.params['model_inh_neuron'] = 'iaf_cond_exp_bias'
#        self.params['model_exc_neuron'] = 'iaf_cond_alpha_bias'
#        self.params['model_inh_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_msn_d1'] = 5
        self.params['num_msn_d2'] = 5
        self.params['n_cells_d1'] = self.params['num_msn_d1'] * self.params['n_actions']
        self.params['n_cells_d2'] = self.params['num_msn_d2'] * self.params['n_actions']
#        self.params['param_msn_d1'] = { 't_ref': 2.0, 'V_reset':-80., 'tau_syn_ex': 5., 'tau_syn_in' : 5.,  \
#                'g_L':16.667, 'C_m':250., 'E_L':-70., 'E_in': -70.}
        self.params['param_msn_d1'] = {'fmax':self.params['fmax'], 'tau_j': self.tau_j, 'tau_e': self.tau_e,\
                'tau_p': self.tau_p, 'epsilon': self.epsilon, 't_ref': 2.0, 'gain': self.params['gain'], \
                'V_reset':-80., 'tau_syn_ex': 5., 'tau_syn_in' : 5., 
                'g_L':16.667, 'C_m':250., 'E_L':-70., 'E_in': -70., \
                'K': self.K, 'gain': self.params['gain']}
#                'g_L': 50., 'C_m':250., 'E_L':-70., 'E_in': -70., \
        self.params['param_msn_d2'] = self.params['param_msn_d1'].copy()
        # old params for alpha shaped synapses
#        self.params['param_msn_d1'] = {'fmax':self.params['fmax'], 'tau_j': self.tau_j, 'tau_e': self.tau_e,\
#                'tau_p': self.tau_p, 'epsilon': self.epsilon, 't_ref': 2.0, 'gain': self.params['gain'], \
#                'V_reset':-70., 'tau_syn_ex': .5, 'tau_syn_in' : 5.}
        # iaf_cond_exp_bias default parameters: 
        # {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, 'E_in': -85.0, 'K': 1.0, 'V_m': -70.0, 'V_reset': -60.0, 'V_th': -55.0, 'epsilon': 0.001, 'fmax': 20.0, 'g_L': 16.6667, 'gain': 1.0, 't_ref': 2.0, 'tau_e': 100.0, 'tau_j': 10.0, 'tau_p': 1000.0, 'tau_syn_ex': 0.2, 'tau_syn_in': 2.0 }
        
        ## Output GPi/SNr
        self.params['model_bg_output_neuron'] = 'iaf_cond_exp'
        self.params['num_actions_output'] = 5
        self.params['param_bg_output'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
                'E_in': -80.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -80.0, 'V_th': -50.0, \
                'g_L': 25.0, 't_ref': 1.0, 'tau_syn_ex': 5.0, 'tau_syn_in': 5.0}
        #{'V_reset': -70.0} # to adapt parms to aif_cond_alpha neuron model

        
        ## RP and REWARD
        self.params['model_rp_neuron'] = 'iaf_cond_alpha_bias'
        self.params['num_rp_neurons'] = 15
        self.params['param_rp_neuron'] = {'fmax': self.params['fmax'], 'tau_i':self.tau_i, 'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p':100000., 'epsilon': 0.01, 't_ref': 2., 'gain': 0.}

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


        self.params['mpn_d1_synapse_model'] = 'bcpnn_synapse'
        self.params['mpn_d2_synapse_model'] = 'bcpnn_synapse'
        ## D1 - D1 connections
        # if bcpnn_synapse is used
        self.params['synapse_d1_d1'] = 'bcpnn_synapse'
        self.params['synapse_d2_d2'] = 'bcpnn_synapse'
        bcpnn_init = 0.01
        self.params['bcpnn_init_pi'] = bcpnn_init
        bcpnn_init = self.params['bcpnn_init_pi'] 
        self.params['params_synapse_d1_d1'] = {'p_i': bcpnn_init , 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': self.params['gain'], 'K': self.K, \
                'fmax': self.params['fmax'], 'epsilon': self.epsilon, 'delay': self.params['delay_d1_d1'], \
                'tau_i': self.tau_i, 'tau_j': self.tau_j, 'tau_e': self.tau_e, 'tau_p': self.tau_p}
        self.params['params_synapse_d2_d2'] = {'p_i': bcpnn_init , 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': self.params['gain'], 'K': self.K, \
                'fmax': self.params['fmax'], 'epsilon': self.epsilon, 'delay': self.params['delay_d2_d2'], \
                'tau_i': self.tau_i, 'tau_j': self.tau_j, 'tau_e': self.tau_e, 'tau_p': self.tau_p}



        self.params['actions_rp'] = 'bcpnn_synapse'
        self.params['param_actions_rp'] = {'p_i': bcpnn_init, 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': self.params['gain'], 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        self.params['states_rp'] = 'bcpnn_synapse'
        self.params['param_states_rp'] = {'p_i': bcpnn_init, 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': self.params['gain'], 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}

        self.params['bcpnn'] = 'bcpnn_synapse'
        self.params['param_bcpnn'] =  {'p_i': bcpnn_init, 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': self.params['gain'], 'K': self.K,'fmax': self.params['fmax'] ,'epsilon': self.epsilon,'delay':1.0,'tau_i': self.tau_i,'tau_j': self.tau_j,'tau_e': self.tau_e,'tau_p': self.tau_p}
        # during learning gain == 0. K = 1.0 : --> 'offline' learning
        # after learning: gain == 1. K = .0

        #Connections States Actions
        self.params['synapse_d1_MT_BG'] = 'bcpnn_synapse'
        self.params['params_synapse_d1_MT_BG'] = {'p_i': bcpnn_init, 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': self.params['gain'], 'K': self.K, \
                'fmax': self.params['fmax'], 'epsilon': self.epsilon, 'delay':1.0, \
                'tau_i': self.tau_i, 'tau_j': self.tau_j, 'tau_e': self.tau_e, 'tau_p': self.tau_p}
        self.params['synapse_d2_MT_BG'] = 'bcpnn_synapse'
        self.params['params_synapse_d2_MT_BG'] = {'p_i': bcpnn_init, 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': self.params['gain'], 'K': self.K, \
                'fmax': self.params['fmax'], 'epsilon': self.epsilon, 'delay':1.0, \
                'tau_i': self.tau_i, 'tau_j': self.tau_j, 'tau_e': self.tau_e, 'tau_p': self.tau_p}

        #Connections REW to RP, STRD1 and STRD2
        self.params['weight_rew_d1'] = 10.
        self.params['weight_rew_d2'] = 10.
        self.params['delay_rew_d1'] = 1.
        self.params['delay_rew_d2'] = 1.

        self.params['weight_rew_rp'] = 10.
        self.params['delay_rew_rp'] = 1.

        # --> TODO: check which params are (not) used?
        #Spike detectors params, copied from BG branch
        self.params['spike_detector_action'] = {'withgid':True, 'withtime':True}
        self.params['spike_detector_d1'] = {'withgid':True, 'withtime':True}
        self.params['spike_detector_d2'] = {'withgid':True, 'withtime':True}
        self.params['spike_detector_states'] = {'withgid':True, 'withtime':True}
        self.params['spike_detector_efference'] = {'withgid':True, 'withtime':True}
        self.params['spike_detector_rew'] = {'withgid':True, 'withtime':True}
        self.params['spike_detector_rp'] = {'withgid':True, 'withtime':True}

        #Spike detectors params, old from my branch 
        self.params['spike_detector_output_action'] = {'withgid':True, 'withtime':True} 
        self.params['spike_detector_test_rp'] = {'withgid':True, 'withtime':True}
        self.params['spike_detector_supervisor'] = {'withgid':True, 'withtime':True}

        self.params['str_to_output_exc_w'] = 10.
        self.params['str_to_output_inh_w'] = -0.
        self.params['str_to_output_exc_delay'] = 1.
        self.params['str_to_output_inh_delay'] = 1.

        #Supervised Learning
        self.params['supervised_on'] = True
        self.params['with_d2'] = True
        self.params['num_neuron_poisson_supervisor'] = 1
        self.params['num_neuron_poisson_input_BG'] = 10
        self.params['active_supervisor_rate'] = 3000.
        self.params['inactive_supervisor_rate'] = 0.
        self.params['active_poisson_input_rate'] = 20.
        self.params['inactive_poisson_input_rate'] = 2.
        self.params['param_poisson_pop_input_BG'] = {}
        self.params['param_poisson_supervisor'] = {}
        self.params['weight_supervisor_strd1'] = 7.
        self.params['weight_supervisor_strd2'] = 7.
        self.params['delay_supervisor_strd1'] = .1
        self.params['delay_supervisor_strd2'] = .1
        self.params['weight_poisson_input'] = 10.
        self.params['delay_poisson_input'] = 1.
        self.params['are_MT_BG_connected'] = True
        self.params['num_neuron_states'] = 30
        self.params['param_states_pop'] = {} 

        # Reinforcement Learning
        self.params['num_neuron_poisson_efference'] = 40
        self.params['active_full_efference_rate'] = 40.
        self.params['inactive_efference_rate'] = 0.
        self.params['active_poisson_input_rate'] = 50.
        self.params['inactive_poisson_input_rate'] = 0.5
        self.params['supervisor_off'] = 0.
        self.params['active_poisson_rew_rate'] = 70.
        self.params['inactive_poisson_rew_rate'] = 1.
        self.params['param_poisson_efference'] = {}
        self.params['weight_efference_strd1'] = 15.
        self.params['weight_efference_strd2'] = 15.
        self.params['delay_efference_strd1'] = 1.
        self.params['delay_efference_strd2'] = 1.

        


    def set_filenames(self, folder_name=None):
        """
        This funcion is called if no params_fn is passed 
        """

        self.set_folder_names(folder_name)

        self.params['mpn_exc_volt_fn'] = 'mpn_exc_volt_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['mpn_exc_spikes_fn'] = 'mpn_exc_spikes_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['mpn_exc_spikes_fn_merged'] = 'mpn_exc_merged_spikes.dat' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['mpn_inh_volt_fn'] = 'mpn_inh_volt_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['mpn_inh_spikes_fn'] = 'mpn_inh_spikes_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['mpn_inh_spikes_fn_merged'] = 'mpn_inh_merged_spikes.dat' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['training_sequence_fn'] = self.params['parameters_folder'] + 'training_stimuli_parameters.txt'

        # bg files:
        self.params['states_spikes_fn'] = 'states_spikes_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['d1_spikes_fn'] = 'd1_spikes_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['d1_volt_fn'] = 'd1_volt_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['d2_spikes_fn'] = 'd2_spikes_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['d2_volt_fn'] = 'd2_volt_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['bg_volt_fn'] = 'bg_volt_'
        self.params['actions_spikes_fn'] = 'actions_spikes_'
        self.params['actions_volt_fn'] = 'actions_volt_' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['efference_spikes_fn'] = 'efference_spikes_'
        self.params['supervisor_spikes_fn'] = 'supervisor_spikes_'
        self.params['rew_spikes_fn'] = 'rew_spikes_'
        self.params['rew_volt_fn'] = 'rew_volt_'
        self.params['rp_spikes_fn'] = 'rp_spikes_'
        self.params['rp_volt_fn'] = 'rp_volt_'

        # merged filenames: CELLTYPE_merged_VOLT/SPIKES.dat
        # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['states_spikes_fn_merged'] = 'states_merged_spikes' 
        self.params['d1_spikes_fn_merged'] = 'd1_merged_spikes' # each action population prints in its own file
        self.params['d2_spikes_fn_merged'] = 'd2_merged_spikes'
        self.params['actions_spikes_fn_merged_all'] = 'all_action_spikes.dat'
        self.params['d1_spikes_fn_merged_all'] = 'all_d1_spikes.dat'
        self.params['d2_spikes_fn_merged_all'] = 'all_d2_spikes.dat'
        self.params['actions_spikes_fn_merged'] = 'actions_merged_spikes'
        self.params['efference_spikes_fn_merged'] = 'efference_merged_spikes'
        self.params['supervisor_spikes_fn_merged'] = 'supervisor_merged_spikes'
        self.params['rew_spikes_fn_merged'] = 'rew_merged_spikes'
        self.params['rp_spikes_fn_merged'] = 'rp_merged_spikes'
        # voltage files
        self.params['d1_volt_fn_merged'] = 'd1_merged_volt.dat' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['d2_volt_fn_merged'] = 'd2_merged_volt.dat' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['actions_volt_fn_merged'] = 'actions_merged_volt.dat' # data_path is already set to spiketimes_folder --> files will be in this subfolder
        self.params['rew_volt_fn_merged'] = 'rew_merged_volt.dat'
        self.params['rp_volt_fn_merged'] = 'rp_merged_volt.dat'

        # figure filenames
        if self.params['training']:
            self.params['bg_rasterplot_fig'] = self.params['figures_folder'] + 'bg_rasterplot_training.png'
        else:
            self.params['bg_rasterplot_fig'] = self.params['figures_folder'] + 'bg_rasterplot_test.png'

        # some filenames, TODO: check if required
        self.params['bg_action_volt_fn'] = 'bg_action_volt_'
        self.params['bg_spikes_fn'] = 'bg_spikes_'
        self.params['bg_spikes_fn_merged'] = 'bg_merged_spikes.dat'

        self.params['bg_gids_fn'] = self.params['parameters_folder'] + 'bg_cell_gids.json'
        self.params['mpn_gids_fn'] = self.params['parameters_folder'] + 'mpn_cell_gids.json'
        
        # input spike files
        self.params['input_st_fn_mpn'] = self.params['input_folder_mpn'] + 'input_spikes_'
        self.params['input_rate_fn_mpn'] = self.params['input_folder_mpn'] + 'input_rate_'
        self.params['input_nspikes_fn_mpn'] = self.params['input_folder_mpn'] + 'input_nspikes_'
        # tuning properties
        self.params['tuning_prop_exc_fn'] = self.params['parameters_folder'] + 'tuning_prop_exc.txt'
        self.params['tuning_prop_inh_fn'] = self.params['parameters_folder'] + 'tuning_prop_inh.txt'
        self.params['receptive_fields_exc_fn'] = self.params['parameters_folder'] + 'receptive_field_sizes_exc.txt'
        # storage for actions (BG), network states (MPN) and motion parameters (on Retina)
        self.params['rewards_given_fn'] = self.params['data_folder'] + 'rewards_given.txt' # contains (vx, vy, action_index)
        self.params['actions_taken_fn'] = self.params['data_folder'] + 'actions_taken.txt' # contains (vx, vy, action_index)
        self.params['bg_action_bins_fn'] = self.params['data_folder'] + 'bg_actions_bins.txt'
        self.params['network_states_fn'] = self.params['data_folder'] + 'network_states.txt'
        self.params['motion_params_fn'] = self.params['data_folder'] + 'motion_params.txt'
        self.params['supervisor_states_fn'] = self.params['data_folder'] + 'supervisor_states.txt'
        self.params['action_indices_fn'] = self.params['data_folder'] + 'action_indices.txt'
        self.params['motion_params_precomputed_fn'] = self.params['data_folder'] + 'motion_params_precomputed.txt'

        # connection filenames
        self.params['mpn_bgd1_conn_fn_base'] = self.params['connections_folder'] + 'mpn_bg_d1_connections'
        self.params['mpn_bgd2_conn_fn_base'] = self.params['connections_folder'] + 'mpn_bg_d2_connections'
        self.params['mpn_bgd1_merged_conn_fn'] = self.params['connections_folder'] + 'merged_mpn_bg_d1_connections.txt'
        self.params['mpn_bgd2_merged_conn_fn'] = self.params['connections_folder'] + 'merged_mpn_bg_d2_connections.txt'
        self.params['d1_d1_conn_fn_base'] = self.params['connections_folder'] + 'd1_d1_connections'
        self.params['d1_d1_merged_conn_fn'] = self.params['connections_folder'] + 'merged_d1_d1_connections.txt'

        # connection development / tracking 
        self.params['mpn_bgd1_conntracking_fn_base'] = self.params['connections_folder'] + 'mpn_bg_d1_connection_dev_'
        self.params['mpn_bgd2_conntracking_fn_base'] = self.params['connections_folder'] + 'mpn_bg_d2_connection_dev_'
        self.params['mpn_bgd1_merged_conntracking_fn_base'] = self.params['connections_folder'] + 'merged_mpn_bg_d1_connection_dev_'
        self.params['mpn_bgd2_merged_conntracking_fn_base'] = self.params['connections_folder'] + 'merged_mpn_bg_d2_connection_dev_'


        #bias filenames
        self.params['bias_d1_fn_base'] = self.params['connections_folder'] + 'bias_d1_'
        self.params['bias_d2_fn_base'] = self.params['connections_folder'] + 'bias_d2_'
        self.params['bias_d1_merged_fn'] = self.params['connections_folder'] + 'bias_d1_merged.json'
        self.params['bias_d2_merged_fn'] = self.params['connections_folder'] + 'bias_d2_merged.json'

    def set_folder_names(self, folder_name=None):
#        super(global_parameters, self).set_default_foldernames(folder_name)
#        folder_name = 'Results_GoodTracking_titeration%d/' % self.params['t_iteration']

        if folder_name == None:
            if self.params['training']:
                folder_name = 'Training_%s_nStim%dx%d_nactions%d_blur%.2f_taup%d/' % (self.params['sim_id'], \
                        self.params['n_training_cycles'], self.params['n_training_stim_per_cycle'], \
                        self.params['n_actions'], self.params['blur_X'], \
                        self.params['params_synapse_d1_MT_BG']['tau_p'])
            else:
                folder_name = 'Test_%s_%d-%d' % (self.params['sim_id'], self.params['test_stim_range'][0], self.params['test_stim_range'][-1])
                folder_name += '_nStim%dx%d_wampD1%.1f_wampD2%.1f/' % \
                        (self.params['n_training_cycles'], self.params['n_training_stim_per_cycle'], \
                         self.params['mpn_d1_weight_amplification'], self.params['mpn_d2_weight_amplification'])
#                        , self.params['t_iteration'], \
#                                self.params['n_actions'], self.params['d1_d1_weight_amplification_pos'], self.params['d1_d1_weight_amplification_neg'], \
#                                self.params['mpn_bg_bias_amplification'])

        assert(folder_name[-1] == '/'), 'ERROR: folder_name must end with a / '

        self.params['folder_name'] = folder_name

        self.params['parameters_folder'] = '%sParameters/' % self.params['folder_name']
        self.params['figures_folder'] = '%sFigures/' % self.params['folder_name']
        self.params['connections_folder'] = '%sConnections/' % self.params['folder_name']
        self.params['tmp_folder'] = '%stmp/' % self.params['folder_name']
        self.params['data_folder'] = '%sData/' % (self.params['folder_name']) # for storage of analysis results etc
        self.params['input_folder_mpn'] = '%sInputSpikes_MPN/' % (self.params['folder_name'])
        self.params['spiketimes_folder'] = '%sSpikes/' % self.params['folder_name']
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['parameters_folder'], \
                            self.params['figures_folder'], \
                            self.params['tmp_folder'], \
                            self.params['connections_folder'], \
                            self.params['data_folder'], \
                            self.params['spiketimes_folder'], \
                            self.params['input_folder_mpn'] ]

        self.params['params_fn_json'] = '%ssimulation_parameters.json' % (self.params['parameters_folder'])


        self.params['folder_names'].append(self.params['spiketimes_folder'])
        self.params['folder_names'].append(self.params['input_folder_mpn'])
#        self.create_folders()


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


