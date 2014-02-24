import nest
import numpy as np
import utils

class MotionPrediction(object):

    def __init__(self, params, VI, comm=None):
        
        self.VI = VI

        self.pc_id, self.n_proc = nest.Rank(), nest.NumProcesses()
        self.comm = comm # mpi communicator needed to broadcast nspikes between processes
        if comm != None:
            assert (comm.rank == self.pc_id), 'mpi4py and NEST tell me different PIDs!'
            assert (comm.size == self.n_proc), 'mpi4py and NEST tell me different PIDs!'

        self.params = params
        self.current_state = None

        self.list_of_populations = []
        self.local_idx_exc = []
        self.local_idx_inh = []
        self.spike_times_container = []

        self.setup_synapse_types()
        self.create_exc_network()
        self.create_inh_network()
#        self.connect_exc_inh()
        self.record_voltages(self.params['gids_to_record_mpn'])

        nest.SetKernelStatus({'data_path':self.params['spiketimes_folder_mpn'], 'overwrite_files': True})
#        print 'DEBUG pid %d has local_idx_exc:' % (self.pc_id), self.local_idx_exc
#        print 'DEBUG pid %d has local_idx_inh:' % (self.pc_id), self.local_idx_inh

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
            nest.SetStatus([self.stimulus[i_]], {'spike_times' : stim[i_]})
#            print 't_current: %d udpating input stimulus spiketrains:' % self.t_current, i_, stim[i_]


    def create_exc_network(self, dummy=False):

        if dummy:
            self.create_dummy_network()
            # record spikes
            self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':self.params['mpn_exc_spikes_fn']})
            for state in xrange(self.params['n_states']):
                nest.ConvergentConnect(self.list_of_populations[state], self.exc_spike_recorder)

        else:

            cell_params = self.params['cell_params_exc_mpn'].copy()

            self.exc_pop = nest.Create(self.params['neuron_model_mpn'], self.params['n_exc_mpn'], params=cell_params)
            self.list_of_populations.append(self.exc_pop)
            self.local_idx_exc += self.get_local_indices(self.exc_pop) # get the GIDS of the neurons that are local to the process

            self.stimulus = nest.Create('spike_generator', self.n_local_exc)
            # connect stimuli containers to the local cells
            for i_, gid in enumerate(self.local_idx_exc):
                nest.Connect([self.stimulus[i_]], [self.exc_pop[gid - 1]], model='input_exc_0')




            self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':self.params['mpn_exc_spikes_fn']})
            nest.ConvergentConnect(self.exc_pop, self.exc_spike_recorder)



    def create_inh_network(self):

        cell_params = self.params['cell_params_inh_mpn'].copy()

        self.inh_pop = nest.Create(self.params['neuron_model_mpn'], self.params['n_inh_mpn'], params=cell_params)
        self.list_of_populations.append(self.inh_pop)
        self.local_idx_inh += self.get_local_indices(self.inh_pop) # get the GIDS of the neurons that are local to the process

        self.inh_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':self.params['mpn_inh_spikes_fn']})
        nest.ConvergentConnect(self.inh_pop, self.inh_spike_recorder)



    def connect_exc_inh(self):

#        nest.RandomConvergentConnect(self.exc_pop, self.inh_pop, self.params['n_ee_mpn'], weeght=self.params['w_ee_mpn'], delay=self.params['delay_ee_mpn'], model='static_synapse')
        nest.RandomConvergentConnect(self.exc_pop, self.inh_pop, self.params['n_ei_mpn'], weight=self.params['w_ei_mpn'], delay=self.params['delay_ei_mpn'], model='static_synapse')
        nest.RandomConvergentConnect(self.inh_pop, self.exc_pop, self.params['n_ie_mpn'], weight=self.params['w_ie_mpn'], delay=self.params['delay_ie_mpn'], model='static_synapse')
        nest.RandomConvergentConnect(self.inh_pop, self.inh_pop, self.params['n_ii_mpn'], weight=self.params['w_ii_mpn'], delay=self.params['delay_ii_mpn'], model='static_synapse')




    def create_dummy_network(self):

        cell_params = self.params['cell_params_exc_mpn'].copy()

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


    def get_current_state(self, tuning_prop_exc):
        """
        This function should return an integer between 0 and params['n_states']
        based on the spiking activity of the network during the last iteration (t_iteration [ms]).
        """
        all_events = nest.GetStatus(self.exc_spike_recorder)[0]['events']
        recent_event_idx = all_events['times'] > self.t_current
        new_event_times = all_events['times'][recent_event_idx]
        new_event_gids = all_events['senders'][recent_event_idx]

        if self.comm != None:
            gids_spiked, nspikes = utils.communicate_local_spikes(new_event_gids, self.comm)
        else:
            gids_spiked = new_event_gids.unique() - 1
            nspikes = np.zeros(len(new_event_gids))
            for i_, gid in enumerate(new_event_gids):
                nspikes[i_] = (new_event_gids == gid).nonzero()[0].size
        
        # for all local gids: count occurence in new_event_gids
        stim_params_readout = self.readout_spiking_activity(tuning_prop_exc, gids_spiked, nspikes)
        self.t_current += self.params['t_iteration']
        return stim_params_readout


    def readout_spiking_activity(self, tuning_prop, gids, nspikes):

        if len(gids) == 0:
            print '\nWARNING:\n\tNo spikes on core %d emitted!!!\n\tMotion Prediction Network was silent!\nReturning nonevalid stimulus prediction\n' % (self.pc_id)
            return [0, 0, 0, 0]

        confidence = nspikes / float(nspikes.sum())
        n_dim = tuning_prop[0, :].size
        prediction = np.zeros(n_dim)
        for i_, gid in enumerate(gids):
            prediction += tuning_prop[gid, :] * confidence[i_]
        return prediction

        


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

        if gids_to_record == 'random':
            gids_to_record = np.random.randint(1, self.params['n_exc_mpn'], self.params['n_exc_to_record_mpn'])
        elif gids_to_record == None:
            gids_to_record = self.VI.get_gids_near_stim_trajectory()[:self.params['n_exc_to_record_mpn']]
#            gids_to_record = self.VI.get_gids_near_stim_trajectory(verbose=self.params['debug_mpn'])[:self.params['n_exc_to_record_mpn']]

        if self.pc_id == 0:
            np.savetxt(self.params['gids_to_record_fn_mp'], gids_to_record)
        self.voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.2})
        nest.SetStatus(self.voltmeter,[{"to_file": True, "withtime": True, 'label' : self.params['mpn_exc_volt_fn']}])
            
        nest.ConvergentConnect(self.voltmeter, gids_to_record)
#        nest.DivergentConnect(self.voltmeter, gids_to_record)
#        for gid in gids_to_record:
#            if gid in self.local_idx_exc:

#                if dummy:
#                    mc_idx, idx_in_mc = self.get_indices_for_gid(gid)
#                    nest.ConvergentConnect(self.voltmeter, [self.list_of_populations[mc_idx][idx_in_mc]])
#                else:
#                    nest.ConvergentConnect(self.voltmeter, [self.exc_pop[gid - 1]])


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

