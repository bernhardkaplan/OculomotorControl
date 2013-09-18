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
        self.create_network()
        self.record_voltages(self.params['gids_to_record_mpn'])

        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder_mpn'], 'overwrite_files': True})
        print 'DEBUG pid %d has local_idx_exc:' % (self.pc_id), self.local_idx_exc

        self.t_current = 0

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


    def create_network(self, dummy=False):

        if dummy:
            self.create_dummy_network()
            # record spikes
            self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
            for state in xrange(self.params['n_states']):
                nest.ConvergentConnect(self.list_of_populations[state], self.exc_spike_recorder)

        else:
            self.create_exc_pop()
            self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
            nest.ConvergentConnect(self.exc_pop, self.exc_spike_recorder)


    def create_exc_pop(self):
        cell_params = self.params['cell_params_mpn'].copy()

        self.exc_pop = nest.Create(self.params['neuron_model_mpn'], self.params['n_exc_mpn'], params=cell_params)
        self.list_of_populations.append(self.exc_pop)
        self.local_idx_exc += self.get_local_indices(self.exc_pop) # get the GIDS of the neurons that are local to the process

        self.stimulus = nest.Create('spike_generator', self.n_local_exc)
        # connect stimuli containers to the local cells
        for i_, gid in enumerate(self.local_idx_exc):
            print 'debug', i_, gid, len(self.exc_pop), self.pc_id
            nest.Connect([self.stimulus[i_]], [self.exc_pop[gid - 1]], model='input_exc_0')



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


    def get_current_state(self):
        """
        This function should return an integer between 0 and params['n_states']
        based on the spiking activity of the network during the last iteration (t_iteration [ms]).
        """
        all_events = nest.GetStatus(self.exc_spike_recorder)[0]['events']
        recent_event_idx = all_events['times'] > self.t_current
        new_event_times = all_events['times'][recent_event_idx]
        new_event_gids = all_events['senders'][recent_event_idx]
        state_activity = np.array(new_event_gids) / self.params['n_exc_per_mc']
#        print 'new_event_times between %d - %d' % (self.t_current, self.t_current + self.params['t_iteration']),  new_event_times
#        print 'new_event_gids', new_event_gids
#        print 'State activity:', state_activity
        cnt, bins = np.histogram(state_activity, bins=range(self.params['n_states']))
        wta_state = np.argmax(cnt)
        self.t_current += self.params['t_iteration']
        return wta_state




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
        

    def record_voltages(self, gids_to_record=None, dummy=False):

        if gids_to_record == None:
            gids_to_record = np.random.randint(1, self.params['n_cells_mpn'], self.params['n_cells_to_record_mpn'])
        voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.5})
        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
            
        for gid in gids_to_record:
            if gid in self.local_idx_exc:

                if dummy:
                    mc_idx, idx_in_mc = self.get_indices_for_gid(gid)
                    nest.ConvergentConnect(voltmeter, [self.list_of_populations[mc_idx][idx_in_mc]])
                else:
                    nest.ConvergentConnect(voltmeter, [self.exc_pop[gid - 1]])


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

