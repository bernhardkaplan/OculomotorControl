import os
import numpy as np
import json
import ParameterContainer

class global_parameters(ParameterContainer.ParameterContainer):
    def __init__(self, output_dir=None):
        super(global_parameters, self).__init__(output_dir)
        self.set_default_params()


    def set_default_params(self):

#        super(global_parameters, self).set_default_params()

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_sim'] = 3000.                 # [ms] total simulation time
        self.params['t_integrate'] = 50.             # [ms] stimulus integration time, after this time the input stimulus will be transformed
        self.params['n_iterations'] = int(round(self.params['t_sim'] / self.params['t_integrate']))

        self.set_visual_input_params()


    def set_visual_input_params(self):

        self.params['x_offset'] = 0.
        self.params['v_stim'] = 1.      # [Hz] +1. means it traverses the whole visual field from left to right within 1000 ms
        self.params['dt_stim'] = 1.     # [ms] temporal resolution with which the stimulus trajectory is computed

        # SENSORY LAYER
        self.params['N_RF_X'] = 50
        self.params['N_RF_V'] = 5       # at each position of a spatial receptive field there are this many units with different preferred directions
        self.params['n_exc'] = self.params['N_RF_X'] * self.params['N_RF_V']

        # OUTPUT FILES
#        self.params['trajectory_fn_base'] = '


    def set_filenames(self, folder_name=None):
        folder_name = 'AwesomeResults/'
        super(global_parameters, self).set_filenames(folder_name)
#        print 'Folder names:', self.params['folder_names']
