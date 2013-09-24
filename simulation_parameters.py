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
        super(global_parameters, self).__init__() # call the constructor of the super/mother class
        
        if params_fn == None:
            self.set_default_params()
            self.set_visual_input_params()
            self.set_mpn_params()


    def set_default_params(self):
        """
        Here all the simulation parameters NOT being filenames are set.
        """

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_sim'] = 1000.                 # [ms] total simulation time
        self.params['t_iteration'] = 100.             # [ms] stimulus integration time, after this time the input stimulus will be transformed
        self.params['dt'] = 0.1                      # [ms]
        self.params['n_iterations'] = int(round(self.params['t_sim'] / self.params['t_iteration']))
        self.params['dt_input_mpn'] = 0.1           # [ms] time step for the inhomogenous Poisson process for input spike train generation

        # #####################################
        # CONNECTING MPN --> BG
        # #####################################
        self.params['w_exc_mpn_bg'] = 10.

        # #####################################
        # BASAL GANGLIA PARAMETERS
        # #####################################
        self.params['n_exc_bg'] = 100
        self.params['n_actions'] = 3
        self.params['n_states'] = 10
        self.params['initial_state'] = (.3, .5, -.5, .0) # initial motion parameters: (x, y, v_x, v_y) position and direction at start


    def set_visual_input_params(self):
        """
        Here only those parameters should be set, which have nothing to do
        with other modules or objects
        TODO: move the cell numbers etc to simulation parameters, 
        as it affects how connections are set up between the MotionPrediction and the BasalGanglia module
        """

        self.params['visual_stim_seed'] = 123
        self.params['tuning_prop_seed'] = 0
        self.params['dt_stim'] = 1.     # [ms] temporal resolution with which the stimulus trajectory is computed
        self.params['motion_params'] = [.0, .5 , .5, 0, np.pi/6.0] # (x, y, v_x, v_y, orientation of bar)
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
            self.params['n_theta'] = 1



    def set_mpn_params(self):

        # #####################################
        # MOTION PREDICTION NETWORK PARAMETERS 
        # #####################################
        self.params['neuron_model_mpn'] = 'iaf_cond_exp'
        self.params['cell_params_exc_mpn'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
                'E_in': -85.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -60.0, 'V_th': -55.0, \
                'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 0.2, 'tau_syn_in': 2.0}
        self.params['cell_params_inh_mpn'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
                'E_in': -85.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -60.0, 'V_th': -55.0, \
                'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 0.2, 'tau_syn_in': 2.0}
        # input parameters
        self.params['w_input_exc_mpn'] = 20. # [nS]
        self.params['f_max_stim'] = 5000.       # [Hz] Max rate of the inhomogenous Poisson process


        # ##############################
        # EXCITATORY NETWORK PARAMETERS
        # ##############################
        # network properties, size, number of preferred directions
        self.params['n_v'] = 6
        self.params['n_hc'] = self.params['n_rf_x'] * self.params['n_rf_y']
        self.params['n_mc_per_hc'] = self.params['n_v'] * self.params['n_theta']
        self.params['n_mc'] = self.params['n_hc'] * self.params['n_mc_per_hc']
        self.params['n_exc_per_state'] = 3
        self.params['n_exc_per_mc'] = self.params['n_exc_per_state']
        self.params['n_exc_mpn'] = self.params['n_mc'] * self.params['n_exc_per_mc']
        print 'n_hc: %d\tn_mc_per_hc: %d\tn_mc: %d\tn_exc_per_mc: %d' % (self.params['n_hc'], self.params['n_mc_per_hc'], self.params['n_mc'], self.params['n_exc_per_mc'])
        self.params['gids_to_record_mpn'] = None
        self.params['log_scale'] = 2.0 # base of the logarithmic tiling of particle_grid; linear if equal to one
        self.params['sigma_rf_pos'] = .01 # some variability in the position of RFs
        self.params['sigma_rf_speed'] = .30 # some variability in the speed of RFs
        self.params['sigma_rf_direction'] = .25 * 2 * np.pi # some variability in the direction of RFs
        self.params['sigma_rf_orientation'] = .1 * np.pi # some variability in the direction of RFs
        self.params['n_orientation'] = 1 # number of preferred orientations
        self.params['n_exc_to_record_mpn'] = 10
        self.params['v_max_tp'] = 3.0   # [Hz] maximal velocity in visual space for tuning proprties (for each component), 1. means the whole visual field is traversed within 1 second
        self.params['v_min_tp'] = 0.10  # [a.u.] minimal velocity in visual space for tuning property distribution
        self.params['blur_X'], self.params['blur_V'] = .20, .20
        self.params['blur_theta'] = 1.0
        self.params['torus_width'] = 1.
        self.params['torus_height'] = 1.

        # ##############################
        # INHIBITOTY NETWORK PARAMETERS
        # ##############################
        self.params['fraction_inh_cells_mpn'] = 0.20 # fraction of inhibitory cells in the network, only approximately!
        self.params['n_theta_inh'] = self.params['n_theta']
        self.params['n_v_inh'] = self.params['n_v']
        self.params['n_rf_inh'] = int(round(self.params['fraction_inh_cells_mpn'] * self.params['n_rf']))
        self.params['n_rf_x_inh'] = np.int(np.sqrt(self.params['n_rf_inh'] * np.sqrt(3)))
        # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of n_rf dots?"
        self.params['n_rf_y_inh'] = np.int(np.sqrt(self.params['n_rf_inh'])) 
        self.params['n_inh_mpn' ] = self.params['n_rf_x_inh'] * self.params['n_rf_y_inh'] * self.params['n_theta_inh'] * self.params['n_v_inh'] * self.params['n_orientation'] * self.params['n_exc_per_mc']
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
#        self.params['w_ei_mpn'] = 10.  # [nS]
#        self.params['w_ie_mpn'] = 40. # [nS]
#        self.params['w_ii_mpn'] = 1.  # [nS]
        self.params['w_ei_mpn'] = 0.  # [nS]
        self.params['w_ie_mpn'] = 0. # [nS]
        self.params['w_ii_mpn'] = 0.  # [nS]

        # number of connections to be received by one cell, to get the total number of connections in the network--> multiply by n_tgt_celltype
        self.params['n_ee_mpn'] = int(round(self.params['p_ee_mpn'] * self.params['n_exc_mpn'])) 
        self.params['n_ei_mpn'] = int(round(self.params['p_ei_mpn'] * self.params['n_exc_mpn'])) 
        self.params['n_ie_mpn'] = int(round(self.params['p_ie_mpn'] * self.params['n_inh_mpn'])) 
        self.params['n_ii_mpn'] = int(round(self.params['p_ii_mpn'] * self.params['n_inh_mpn'])) 

        self.params['delay_ee_mpn'] = 1. # [ms]
        self.params['delay_ei_mpn'] = 1. # [ms]
        self.params['delay_ie_mpn'] = 1. # [ms]
        self.params['delay_ii_mpn'] = 1. # [ms]




    def set_filenames(self, folder_name=None):
        """
        This funcion is called if no params_fn is passed 
        """
        folder_name = 'Results_dev/' # this is the main folder containing all information related to one simulation

        super(global_parameters, self).set_filenames(folder_name)
        self.set_folder_names()
        self.params['exc_volt_fn_mpn'] = 'mpn_exc_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['exc_spikes_fn_mpn'] = 'mpn_exc_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['merged_exc_spikes_fn_mpn'] = 'mpn_merged_exc_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['inh_volt_fn_mpn'] = 'mpn_inh_volt_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['inh_spikes_fn_mpn'] = 'mpn_inh_spikes_' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder
        self.params['merged_inh_spikes_fn_mpn'] = 'mpn_merged_inh_spikes.dat' # data_path is already set to spiketimes_folder_mpn --> files will be in this subfolder

        # input spike files
        self.params['input_st_fn_mpn'] = self.params['input_folder_mpn'] + 'input_spikes_'
        self.params['input_rate_fn_mpn'] = self.params['input_folder_mpn'] + 'input_rate_'
        self.params['input_nspikes_fn_mpn'] = self.params['input_folder_mpn'] + 'input_nspikes_'
        # tuning properties
        self.params['tuning_prop_exc_fn'] = self.params['parameters_folder'] + 'tuning_prop_exc.txt'
        self.params['tuning_prop_inh_fn'] = self.params['parameters_folder'] + 'tuning_prop_inh.txt'

        self.params['actions_taken_fn'] = self.params['data_folder'] + 'actions_taken.txt'
        self.params['network_states_fn'] = self.params['data_folder'] + 'network_states.txt'

    def set_folder_names(self):
        self.params['input_folder_mpn'] = '%sInputSpikes_MPN/' % (self.params['folder_name'])
        self.params['spiketimes_folder_mpn'] = '%sSpikes/' % self.params['folder_name']
        self.params['folder_names'].append(self.params['spiketimes_folder_mpn'])
        self.params['folder_names'].append(self.params['input_folder_mpn'])
        self.create_folders()

