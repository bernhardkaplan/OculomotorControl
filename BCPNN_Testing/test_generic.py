import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../PlottingScripts/")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from plot_bcpnn_traces import TracePlotter
import BCPNN
import nest
import numpy as np 
import pylab


def plot_voltages(volt_recorder):
    volt_data = nest.GetStatus(volt_recorder)[0]['events']
    gids = volt_data['senders']
    idx_0 = (gids == 1).nonzero()[0]
    idx_1 = (gids == 2).nonzero()[0]
    volt_0 = volt_data['V_m'][idx_0]
    volt_1 = volt_data['V_m'][idx_1]
    times_0 = volt_data['times'][idx_0]
    times_1 = volt_data['times'][idx_1]

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(times_0, volt_0, label='pre neuron')
    ax.plot(times_1, volt_1, label='post neuron')

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    pylab.legend()


def compute_bcpnn_traces(spike_train_0, spike_train_1, K_vec, syn_params, t_sim, plot=True):
    #######################
    # OFFLINE COMPUTATION
    #######################
    s_pre = BCPNN.convert_spiketrain_to_trace(spike_train_0, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(spike_train_1, t_sim)
    bcpnn_traces = []
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, syn_params, K_vec=K_vec)
    print 'OFFLINE weight :', wij[-1]
    bcpnn_traces = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    if plot:
        dt = 0.1
        TP = TracePlotter({})
        TP.plot_trace_with_spikes(bcpnn_traces, syn_params, dt, output_fn=None, fig=None, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
                extra_txt=None)

#    TP.plot_trace(self, bcpnn_traces, bcpnn_params, dt, output_fn=None, info_txt=None, fig=None, \
#            color_pre='b', color_post='g', color_joint='r', style_joint='-')



class TestBcpnn(object):

    def __init__(self):
        pass

    def init_sim(self, K):
        try:
        #    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
            nest.Install('pt_module')
        except:
            nest.Install('pt_module')

        self.K_values = [K]
        self.times_K_changed = [0.]
        nest.SetKernelStatus({'overwrite_files': True, \
                'resolution' : 0.1})
        cell_params = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
            'E_in': -80.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -80.0, 'V_th': -60.0, \
            'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 5.0, 'tau_syn_in': 5.0}
        self.nrns = nest.Create('iaf_cond_exp_bias', 2, params=cell_params)
        self.voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.1})
        nest.SetStatus(self.voltmeter, [{"to_file": True, "withtime": True, 'label' : 'volt'}])
        nest.ConvergentConnect(self.voltmeter, self.nrns)
        self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'spikes'})
        nest.ConvergentConnect(self.nrns, self.exc_spike_recorder)
        self.source_0 = nest.Create('spike_generator', 1)
        self.source_1 = nest.Create('spike_generator', 1)
        weight = 15.
        nest.Connect(self.source_0, [self.nrns[0]], weight, 1., model='static_synapse')
        nest.Connect(self.source_1, [self.nrns[1]], weight, 1., model='static_synapse')

        # neuron 0 -> neuron 1
        tau_i = 5.
        tau_j = 5.
        tau_e = 300.
        tau_p = 50000.
        fmax = 200.
        epsilon = 1. / (fmax * tau_p)
        bcpnn_init = 0.01
        p_i = bcpnn_init
        p_j = bcpnn_init
        weight = 0.
        p_ij = np.exp(weight) * p_i * p_j
        self.syn_params  = {'p_i': p_i, 'p_j': p_j, 'p_ij': p_ij, 'gain': 1.0, 'K': K, \
                'fmax': fmax, 'epsilon': epsilon, 'delay':1.0, \
                'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
        print 'syn_params init:', self.syn_params
        nest.Connect([self.nrns[0]], [self.nrns[1]], self.syn_params, model='bcpnn_synapse')

        self.t_sim = 0.

    def set_spiking_behavior(self, pre_spikes=True, post_spikes=True, t_start=0., t_stop=100., dt=1., n_rnd_spikes=5):

        # pre synaptic spikes
        if pre_spikes:
            spike_times_0 = np.round((t_stop - t_start) * np.random.random(n_rnd_spikes) + t_start, decimals=1) # spike_times_0 = np.arange(t_start, t_stop, n_rnd_spikes)
            spike_times_0 = np.sort(spike_times_0)
        else:
            spike_times_0 = np.array([])
        if post_spikes: 
            spike_times_1 = np.round((t_stop - t_start) * np.random.random(n_rnd_spikes) + t_start, decimals=1)
            spike_times_1 = np.sort(spike_times_1)
        else:
            spike_times_1 = np.array([])

        print 'DEBUG st 0', spike_times_0
        print 'DEBUG st 1', spike_times_1
        # post synaptic spikes 
        nest.SetStatus(self.source_0, {'spike_times' : spike_times_0})
        nest.SetStatus(self.source_1, {'spike_times' : spike_times_1})


    def get_nest_weight(self):
        conns = nest.GetConnections([self.nrns[0]], [self.nrns[1]], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
        if len(conns) > 0:
            cp = nest.GetStatus([conns[0]])  # retrieve the dictionary for this connection
            pi = cp[0]['p_i']
            pj = cp[0]['p_j']
            pij = cp[0]['p_ij']
            w = np.log(pij / (pi * pj))
            print 'weight after simulation:', w
        return w

    def trigger_pre_spike_and_get_nest_weight(self):
        spike_times_0 = [self.t_sim + .1]
        print 'DEEEBUG', spike_times_0
        nest.SetStatus(self.source_0, {'spike_times' : spike_times_0})
        nest.Simulate(6.1)
        self.t_sim += 6.1

        conns = nest.GetConnections([self.nrns[0]], [self.nrns[1]], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
        if len(conns) > 0:
            cp = nest.GetStatus([conns[0]])  # retrieve the dictionary for this connection
            pi = cp[0]['p_i']
            pj = cp[0]['p_j']
            pij = cp[0]['p_ij']
            w = np.log(pij / (pi * pj))
            print 'weight after simulation:', w
        return w


    def get_spike_trains(self):
        """
        Returns the spike trains for the two neurons
        """
        spike_data = nest.GetStatus(self.exc_spike_recorder)[0]['events']
        gids = spike_data['senders']
        idx_0 = (gids == 1).nonzero()[0]
        st_0 = spike_data['times'][idx_0]
        idx_1 = (gids == 2).nonzero()[0]
        st_1 = spike_data['times'][idx_1]
        print 'spike_times pre:', st_0
        print 'spike_times post:', st_1
        return st_0, st_1

    def simulate(self, t_sim):
        nest.Simulate(t_sim)
        self.t_sim += t_sim

    def set_kappa(self, K, gain=0.):
        self.K_values.append(K)
        self.times_K_changed.append(self.t_sim)
        nest.SetStatus(nest.GetConnections([self.nrns[0]], [self.nrns[1]]), {'K': K, 'gain': gain, 't_k': nest.GetKernelStatus()['time']})


    def get_K_vec(self, dt=0.1):
        K_vec = self.K_values[0] * np.ones(self.t_sim / dt + 1)
        if len(self.times_K_changed) > 1:
            for i_, t_change in enumerate(self.times_K_changed[1:]):
                idx = int(t_change / dt)
                K_vec[idx:] = self.K_values[i_ + 1]
        return K_vec


if __name__ == '__main__':

    np.random.seed(0)
    t_sim = 100
    K = 1.0
    Tester = TestBcpnn()
    Tester.init_sim(K=K)
    Tester.set_kappa(K=0.)
    Tester.set_spiking_behavior(pre_spikes=True, post_spikes=True, t_start=10, t_stop=t_sim-10.)
    Tester.simulate(t_sim)
    Tester.get_nest_weight()
    Tester.trigger_pre_spike_and_get_nest_weight()
    plot_voltages(Tester.voltmeter)
    st_0, st_1 = Tester.get_spike_trains()
    K_vec = Tester.get_K_vec()
    compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim)

    pylab.show() 


    # TODO: create different test cases and for each test case
    # run the varies options:
    #  init_K (K=0) (K=1)
    #  change_K (K --> K) K remains the same
    #  change_K (K --> not K) K is set to the opposite (from 0 --> 1 or the other way around)
    #  change_K (K --> K + .5) change to a different value 
    #  sim with / without pre- and/or post firing
    """
    init_K
    sim
    """

    """
    init_K
    sim
    change_K 
    sim
    """

    """
    init_K
    change_K
    sim
    change_K
    """

    """
    init_K
    change_K
    sim
    change_K
    sim
    """

    """
    init_K
    change_K
    sim
    change_K
    change_K
    sim
    """

    """
    init_K
    change_K
    change_K
    sim
    change_K
    sim
    """


