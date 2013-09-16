import os
import numpy as np
import json
import ParameterContainer

class global_parameters(ParameterContainer.ParameterContainer):
    """
    The parameter class storing the simulation parameters 
    is derived from the ParameterContainer class.
    """

    def __init__(self, params_fn=None):#, output_dir=None):
        """
        Keyword arguments:
        params_fn -- string, if None: set_filenames and set_default_params will be called
        """
        super(global_parameters, self).__init__() # call the constructor of the super/mother class
        
        if params_fn == None:
            self.set_default_params()


    def set_default_params(self):
        """
        Here all the simulation parameters NOT being filenames are set.
        """

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_sim'] = 3000.                 # [ms] total simulation time
        self.params['t_iteration'] = 50.             # [ms] stimulus integration time, after this time the input stimulus will be transformed
        self.params['dt'] = 0.1                      # [ms]
        self.params['n_iterations'] = int(round(self.params['t_sim'] / self.params['t_iteration']))


        # #####################################
        # MOTION PREDICTION NETWORK PARAMETERS 
        # #####################################
        self.params['n_exc_mpn'] = 100
        self.params['neuron_model_mpn'] = 'iaf_cond_exp'
        self.params['cell_params_mpn'] = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
                'E_in': -85.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -60.0, 'V_th': -55.0, \
                'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 0.2, 'tau_syn_in': 2.0}
        self.params['w_input_exc_mpn'] = 50. # [nS]


    def set_filenames(self, folder_name=None):
        """
        This funcion is called if no params_fn is passed 
        """
        folder_name = 'AwesomeResults3/' # this is the main folder containing all information related to one simulation

        super(global_parameters, self).set_filenames(folder_name)
        self.set_folder_names()



    def set_folder_names(self):
        self.params['spiketimes_folder_mpn'] = '%sSpikes/' % self.params['folder_name']
        self.params['folder_names'].append(self.params['spiketimes_folder_mpn'])
        self.create_folders()

