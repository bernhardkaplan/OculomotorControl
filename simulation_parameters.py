import os
import numpy as np
import json
import ParameterContainer

class global_parameters(ParameterContainer.ParameterContainer):
    def __init__(self, params_fn=None, output_dir=None):
        super(global_parameters, self).__init__(output_dir)
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



    def set_filenames(self, folder_name=None):
        folder_name = 'AwesomeResults/'
        super(global_parameters, self).set_filenames(folder_name)
#        print 'Folder names:', self.params['folder_names']
