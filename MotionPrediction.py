import nest


class MotionPrediction(object):

    def __init__(self, params):
        

        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.params = params
        self.current_state = None
        self.setup_synapse_types()
        self.create_network(dummy=True)
        self.create_input_to_network()
        self.record_voltages()

        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder_mpn'], 'overwrite_files': True})

    def setup_synapse_types(self):

        nest.CopyModel('static_synapse', 'input_exc_0', \
                {'weight': self.params['w_input_exc_mpn'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
        nest.CopyModel('static_synapse', 'input_exc_1', \
                {'weight': self.params['w_input_exc_mpn'], 'receptor_type': 1})

        if (not 'bcpnn_synapse' in nest.Models('synapses')):
            nest.Install('pt_module')


    def update_input(self, stim):
        """
        Keyword arguments:
        stim -- list of spike trains with length = self.params['n_exc_mpn']

        """

        self.spike_times_container = 
        for i_, gid in enumerate(self.local_idx_exc):
            spike_times = self.spike_times_container[i_]
            nest.SetStatus([self.stimulus[i_]], {'spike_times' : spike_times})


    def create_network(self, dummy=True):

        if dummy:
            self.create_dummy_network()
        else:
            print 'MotionPrediction.create_network: \tNot yet implemented ... :( \n will now quit'
            exit(1)


    def create_dummy_network(self):

        cell_params = self.params['cell_params_mpn'].copy()
        self.pop_exc = nest.Create(self.params['neuron_model_mpn'], self.params['n_exc_mpn'], params=cell_params)
        # get the GIDS of the neurons that are local to the process
        self.local_idx_exc = self.get_local_indices(self.pop_exc)
        self.spike_times_container = [ [] for i in xrange(self.n_local_exc)]



    def create_input_to_network(self):
        self.stimulus = nest.Create('spike_generator', self.n_local_exc)
        for i_, gid in enumerate(self.local_idx_exc):
            nest.Connect([self.stimulus[i_]], [self.pop_exc[i_]], model='input_exc_0')


    def get_local_indices(self, pop):
        """
        Returns the GIDS assigned to the process.
        """
        local_nodes = []
        node_info   = nest.GetStatus(pop)
        for i_, d in enumerate(node_info):
            if d['local']:
                local_nodes.append(d['global_id'])
        self.n_local_exc = len(local_nodes)
        return local_nodes
        

    def record_voltages(self):

        voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.5})
        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
            
        exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        
        nest.ConvergentConnect(self.pop_exc, exc_spike_recorder)
        nest.ConvergentConnect(voltmeter, self.pop_exc)
#        for i_, gid in enumerate(self.local_idx_exc):
#            nest.Connect([self.pop_exc[i_]], [exc_spike_recorder])

#            nest.ConvergentConnect(voltmeter, [self.list_of_populations[hc_idx][mc_idx_in_hc][idx_in_mc]])
