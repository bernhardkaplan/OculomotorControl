import nest
import numpy as np


class MotionPrediction(object):

    def __init__(self, params):
        

        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.params = params
        self.current_state = None

        self.list_of_populations = []
        self.local_idx_exc = []
        self.spike_times_container = []

        self.setup_synapse_types()
        self.create_network(dummy=True)
        self.record_voltages(self.params['gids_to_record_mpn'])

        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder_mpn'], 'overwrite_files': True})
        print 'DEBUG pid %d has local_idx_exc:' % (self.pc_id), self.local_idx_exc


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
        for i_, gid in enumerate(self.local_idx_exc):
            if len(stim[i_]) > 0:
                nest.SetStatus([self.stimulus[i_]], {'spike_times' : stim[i_]})


    def create_network(self, dummy=True):

        if dummy:
            self.create_dummy_network()
        else:
            print 'MotionPrediction.create_network: \tNot yet implemented ... :( \n will now quit'
            exit(1)

        # record spikes
        exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        for state in xrange(self.params['n_states']):
            nest.ConvergentConnect(self.list_of_populations[state], exc_spike_recorder)


    def create_dummy_network(self):

        cell_params = self.params['cell_params_mpn'].copy()

        for state in xrange(self.params['n_states']):
            pop = nest.Create(self.params['neuron_model_mpn'], self.params['n_exc_per_mc'], params=cell_params)
            self.list_of_populations.append(pop)
            self.local_idx_exc += self.get_local_indices(pop) # get the GIDS of the neurons that are local to the process

        self.n_local_exc = len(self.local_idx_exc)
        self.stimulus = nest.Create('spike_generator', self.n_local_exc)
        # connect stimuli containers to the local cells
        for i_ in xrange(self.n_local_exc):
            gid = self.local_idx_exc[i_]
            mc_idx = (gid - 1) / self.params['n_exc_per_mc']
            idx_in_mc = (gid - 1) - mc_idx * self.params['n_exc_per_mc']
            nest.Connect([self.stimulus[i_]], [self.list_of_populations[mc_idx][idx_in_mc]], model='input_exc_0')

        self.spike_times_container = [ [] for i in xrange(len(self.local_idx_exc))]



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
        

    def record_voltages(self, gids_to_record=None):

        if gids_to_record == None:
            gids_to_record = np.random.randint(1, self.params['n_cells_mpn'], self.params['n_cells_to_record_mpn'])

        voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.5})
        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
            
        for gid in gids_to_record:
            if gid in self.local_idx_exc:
                mc_idx, idx_in_mc = self.get_indices_for_gid(gid)
                nest.ConvergentConnect(voltmeter, [self.list_of_populations[mc_idx][idx_in_mc]])


    def get_indices_for_gid(self, gid):
        """Returns the HC, MC, and within MC index for the gid
        """

        # without hypercolumns, i.e. with MCs (=states) only
        mc_idx = (gid - 1) / self.params['n_exc_per_mc']
        idx_in_mc = (gid - 1) - mc_idx * self.params['n_exc_per_mc']
        return mc_idx, idx_in_mc

        # with hypercolumns:
#        n_per_hc = self.params['n_mc_per_hc'] * self.params['n_exc_per_mc']
#        mc_idx = (gid - 1) / self.params['n_exc_per_mc']
#        hc_idx = (gid - 1) / n_per_hc
#        mc_idx_in_hc = mc_idx - hc_idx * self.params['n_mc_per_hc']
#        idx_in_mc = (gid - 1) - mc_idx * self.params['n_exc_per_mc']

#        return hc_idx, mc_idx_in_hc, idx_in_mc

