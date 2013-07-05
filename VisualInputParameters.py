import os
import numpy as np
import json
import ParameterContainer

class VisualInputParameters(ParameterContainer.ParameterContainer):
    def __init__(self, params_fn=None):
        super(VisualInputParameters, self).__init__(params_fn)
        if params_fn == None:
            self.set_default_params()


    def set_default_params(self):

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_sim'] = 3000.                 # [ms] total simulation time
        self.params['t_integrate'] = 50.             # [ms] stimulus integration time, after this time the input stimulus will be transformed
        self.params['dt'] = 0.1                      # [ms]
        self.params['n_iterations'] = int(round(self.params['t_sim'] / self.params['t_integrate']))

        self.set_visual_input_params()


    def set_visual_input_params(self):

        self.params['tuning_prop_seed'] = 0
        self.params['x_offset'] = 0.
        self.params['dt_stim'] = 1.     # [ms] temporal resolution with which the stimulus trajectory is computed
        self.params['motion_params'] = [.0, .5 , .5, 0, np.pi/6.0] # (x, y, v_x, v_y, orientation of bar)


        # ###################
        # HEXGRID PARAMETERS
        # ###################
        self.params['n_grid_dimensions'] = 1     # decide on the spatial layout of the network

        self.params['n_rf'] = 30
        if self.params['n_grid_dimensions'] == 2:
            self.params['n_rf_x'] = np.int(np.sqrt(self.params['n_rf'] * np.sqrt(3)))
            self.params['n_rf_y'] = np.int(np.sqrt(self.params['n_rf'])) 
            # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of n_rfdots?"
            self.params['n_theta'] = 3# resolution in velocity norm and direction
        else:
            self.params['n_rf_x'] = 20
            self.params['n_rf_y'] = 1
            self.params['n_theta'] = 1
        self.params['n_v'] = 5
        self.params['n_hc'] = self.params['n_rf_x'] * self.params['n_rf_y']
        self.params['n_mc_per_hc'] = self.params['n_v'] * self.params['n_theta']
        self.params['n_mc'] = self.params['n_hc'] * self.params['n_mc_per_hc']
        self.params['n_exc_per_mc'] = 10
        self.params['n_exc'] = self.params['n_mc'] * self.params['n_exc_per_mc']

        self.params['log_scale'] = 2.0 # base of the logarithmic tiling of particle_grid; linear if equal to one
        self.params['sigma_rf_pos'] = .01 # some variability in the position of RFs
        self.params['sigma_rf_speed'] = .30 # some variability in the speed of RFs
        self.params['sigma_rf_direction'] = .25 * 2 * np.pi # some variability in the direction of RFs
        self.params['sigma_rf_orientation'] = .1 * np.pi # some variability in the direction of RFs
        self.params['n_orientation'] = 1 # number of preferred orientations

        # ###################
        # NETWORK PARAMETERS
        # ###################
        self.params['fraction_inh_cells'] = 0.20 # fraction of inhibitory cells in the network, only approximately!
        self.params['n_theta_inh'] = self.params['n_theta']
        self.params['n_v_inh'] = self.params['n_v']
        self.params['n_rf_inh'] = int(round(self.params['fraction_inh_cells'] * self.params['n_rf']))
        self.params['n_rf_x_inh'] = np.int(np.sqrt(self.params['n_rf_inh'] * np.sqrt(3)))
        # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of n_rf dots?"
        self.params['n_rf_y_inh'] = np.int(np.sqrt(self.params['n_rf_inh'])) 
        self.params['n_inh' ] = self.params['n_rf_x_inh'] * self.params['n_rf_y_inh'] * self.params['n_theta_inh'] * self.params['n_v_inh'] * self.params['n_orientation'] * self.params['n_exc_per_mc']
        self.params['n_cells'] = self.params['n_exc'] + self.params['n_inh']
        print 'n_hc: %d\tn_mc_per_hc: %d\tn_mc: %d\tn_exc_per_mc: %d' % (self.params['n_hc'], self.params['n_mc_per_hc'], self.params['n_mc'], self.params['n_exc_per_mc'])
        print 'n_cells: %d\tn_exc: %d\tn_inh: %d\nn_inh / n_exc = %.3f\tn_inh / n_cells = %.3f' \
                % (self.params['n_cells'], self.params['n_exc'], self.params['n_inh'], \
                self.params['n_inh'] / float(self.params['n_exc']), self.params['n_inh'] / float(self.params['n_cells']))


        self.params['v_max_tp'] = 3.0   # [Hz] maximal velocity in visual space for tuning proprties (for each component), 1. means the whole visual field is traversed within 1 second
        self.params['v_min_tp'] = 0.10  # [a.u.] minimal velocity in visual space for tuning property distribution
        self.params['blur_X'], self.params['blur_V'] = .15, .45
        self.params['blur_theta'] = 1.0
        self.params['torus_width'] = 1.
        self.params['torus_height'] = 1.


    def set_filenames(self, main_folder=None):
        if main_folder == None:
            main_folder = 'AwesomeResults/'
        super(VisualInputParameters, self).set_filenames(main_folder)
#        print 'Folder names:', self.params['folder_names']
