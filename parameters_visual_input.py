import os
import numpy as np
import json
import ParameterContainer

class global_parameters(ParameterContainer.ParameterContainer):
    def __init__(self, output_dir=None):
        super(global_parameters, self).__init__(output_dir)
        self.params = {}
        self.set_default_params()
        self.set_filenames()

    def set_default_params(self):

        self.params['abstract'] = True 

        # ###################
        # SENSORY LAYER SIZE
        # ###################
        self.params['N_RF_X'] = 50  # number of receptive fields on x-axis
        self.params['N_RF_Y'] = 1   # number of RF on y-axis
        self.params['N_V'] = 1      # each spatial RF has this number of cells tuned for N_V different speeds

        # ######################
        # SIMULATION PARAMETERS
        # ######################
        self.params['t_sim'] = 3000.                 # [ms] total simulation time
        self.params['t_integrate'] = 50.             # [ms] stimulus integration time, after this time the input stimulus will be transformed

        self.params['t_stimulus'] = 1000.            # [ms] time for a stimulus of speed 1.0 to cross the whole visual field from 0 to 1.

        if self.params['abstract']:
            self.params['dt_rate'] = 1.                # [ms] time step for the non-homogenous Poisson process
        else:
            self.params['dt_rate'] = .1                # [ms] time step for the non-homogenous Poisson process
        # 5.0 for abstract learning, 0.1 when used as envelope for poisson procees
        self.params['n_gids_to_record'] = 30

        # #############
        # BCPNN PARAMS
        # #############
        tau_p = 1000 
        tau_pij = tau_p
        self.params['tau_dict'] = {'tau_zi' : 50.,    'tau_zj' : 5.,
                                   'tau_ei' : 100.,   'tau_ej' : 100., 'tau_eij' : 100.,
                                   'tau_pi' : tau_p,  'tau_pj' : tau_p, 'tau_pij' : tau_pij,
                                  }

        # ###############
        # SPEED TUNING
        # ###############
        self.params['v_max_tp'] = 3.0  # [a.u.] maximal velocity in visual space for tuning_parameters (for each component), 1. means the whole visual field is traversed within 't_stimulus'
        self.params['v_min_tp'] = 0.15  # [a.u.] minimal velocity in visual space for training
        self.params['v_max_training'] = 0.2
        self.params['v_min_training'] = 0.2

        self.params['blur_X'], self.params['blur_V'] = .15, .15
        # the blur parameter represents the input selectivity:
        # high blur means many cells respond to the stimulus
        # low blur means high input selectivity, few cells respond
        # the maximum number of spikes as response to the input alone is not much affected by the blur parameter

        # ###################
        # TRAINING PARAMETERS
        # ###################
        self.params['n_theta'] = 1 # number of different orientations to train with
        self.params['n_speeds'] = 1
        self.params['n_cycles'] = 1
        self.params['n_stim_per_direction'] = 1 # each direction is trained this many times


    def set_cell_params(self):
        # ###################
        # CELL PARAMETERS   #
        # ###################
        # for later
        self.params['neuron_model'] = 'IF_cond_exp' # alternative: 'EIF_cond_exp_isfa_ista'
        self.params['tau_syn_exc'] = 5.0 # [ms]
        self.params['tau_syn_inh'] = 10.0 # [ms]
        if self.params['neuron_model'] == 'IF_cond_exp':
            self.params['cell_params_exc'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
            self.params['cell_params_inh'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
        elif self.params['neuron_model'] == 'IF_cond_alpha':
            self.params['cell_params_exc'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
            self.params['cell_params_inh'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E': self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70}
        elif self.params['neuron_model'] == 'EIF_cond_exp_isfa_ista':
            self.params['cell_params_exc'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E':self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70., \
                    'b' : 0.2, 'a':1.}
            self.params['cell_params_inh'] = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E':self.params['tau_syn_exc'], 'tau_syn_I':self.params['tau_syn_inh'], 'tau_m' : 10., 'v_reset' : -70., 'v_rest':-70., \
                    'b' : 0.2, 'a':1.}
        self.params['v_init'] = -65.                 # [mV]
        self.params['v_init_sigma'] = 10.             # [mV]


    def set_folder_name(self, folder_name=None):
        """
        Set the root folder name where results will be stored.

        Keyword arguments:
        folder_name -- string
        """

        if folder_name == None:
            folder_name = 'Abstract-OneDim/'
            self.params['folder_name'] = folder_name
        else:
            self.params['folder_name'] = folder_name
        print 'Folder name:', self.params['folder_name']


    def set_filenames(self, folder_name=None):
        """
        Set all filenames and subfolders.

        Keyword arguments
        folder_name -- string
        """

        self.set_folder_name(folder_name)

        self.params['input_folder'] = "%sInputSpikeTrains/"   % self.params['folder_name']# folder containing the input spike trains for the network generated from a certain stimulus
        self.params['spiketimes_folder'] = "%sOutputSpikes/" % self.params['folder_name']
        self.params['volt_folder'] = "%sVoltageTraces/" % self.params['folder_name']
        self.params['parameters_folder'] = "%sParameters/" % self.params['folder_name']
        self.params['connections_folder'] = "%sConnections/" % self.params['folder_name']
        self.params['activity_folder'] = "%sANNActivity/" % self.params['folder_name']
        self.params['bcpnntrace_folder'] = "%sBcpnnTraces/" % self.params['folder_name']
        self.params['figures_folder'] = "%sFigures/" % self.params['folder_name']
        self.params['movie_folder'] = "%sMovies/" % self.params['folder_name']
        self.params['tmp_folder'] = "%stmp/" % self.params['folder_name']
        self.params['data_folder'] = '%sData/' % (self.params['folder_name']) # for storage of analysis results etc
        self.params['training_input_folder'] = "%sTrainingInput/"   % self.params['folder_name'] # folder containing the parameters used for training the network
        self.params['folder_names'] = [self.params['folder_name'], \
                            self.params['spiketimes_folder'], \
                            self.params['volt_folder'], \
                            self.params['parameters_folder'], \
                            self.params['connections_folder'], \
                            self.params['activity_folder'], \
                            self.params['bcpnntrace_folder'], \
                            self.params['figures_folder'], \
                            self.params['movie_folder'], \
                            self.params['tmp_folder'], \
                            self.params['data_folder'], \
                            self.params['training_input_folder'], \
                            self.params['input_folder']] # to be created if not yet existing

        self.params['params_fn_json'] = '%ssimulation_parameters.json' % (self.params['parameters_folder'])


    def create_folders(self):
        """
        Must be called from 'outside' this class before the simulation
        """

        for f in self.params['folder_names']:
            if not os.path.exists(f):
                print 'Creating folder:\t%s' % f
                os.system("mkdir %s" % (f))

    def load_params_from_file(self, fn):
        """
        Loads the file via json from a filename
        Returns the simulation parameters stored in a file 
        Keyword arguments:
        fn -- file name
        """
        f = file(fn, 'r')
        params = json.load(f)
        return params


    def update_values(self, to_update):
        """
        Updates the parameters given in to_update and sets the filenames
        Keyword arguments:
        to_update -- dictionary storing the parameters to be updated
        """
        for key, value in kwargs.iteritems():
            self.params[key] = value
        # update the possibly dependent parameters
        self.set_filenames()


    def write_parameters_to_file(self, fn=None):
        if not (os.path.isdir(self.params['folder_name'])):
            print 'Creating folder:\n\t%s' % self.params['folder_name']
            self.create_folders()
        if fn == None:
            fn = self.params['params_fn_json']
        print 'Writing parameters to: %s' % (fn)
        output_file = file(fn, 'w')
        d = json.dump(self.params, output_file)


class ParameterContainer(parameter_storage):

    def __init__(self, fn):
        super(ParameterContainer, self).__init__()
        self.root_dir = os.path.dirname(fn)
        # If the folder has been moved, all filenames need to be updated
        self.update_values({self.params['folder_name'] : self.root_dir})

    def load_params(self, fn):
        self.params = ntp.ParameterSet(fn)

    def update_values(self, kwargs):
        for key, value in kwargs.iteritems():
            self.params[key] = value
        # update the dependent parameters
        self.ParamSet = ntp.ParameterSet(self.params)

    def create_folders(self):
        """
        Must be called from 'outside' this class before the simulation
        """
        for f in self.params['folder_names']:
            if not os.path.exists(f):
                print 'Creating folder:\t%s' % f
                os.system("mkdir %s" % (f))

    def load_params(self):
        """
        return the simulation parameters in a dictionary
        """
        return self.params


    def write_parameters_to_file(self, fn=None):
        if fn == None:
            fn = self.params['params_fn']
        print 'Writing parameters to: %s' % (fn)
#        if not (os.path.isdir(self.params['folder_name'])):
#            print 'Creating folder:\n\t%s' % self.params['folder_name']
#            os.system('/bin/mkdir %s' % self.params['folder_name'])
        print 'Writing parameters to: %s' % (fn)
        output_file = file(fn, 'w')
        d = json.dump(self.params, output_file)

