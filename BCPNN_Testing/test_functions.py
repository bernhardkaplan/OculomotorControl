"""
TEST 0:

    init_K
    sim

-------------------

TEST 1:

    init_K
    sim
    change_K


-------------------
TEST 2:

    init_K
    change_K 
    sim

-------------------


TEST 3-A:

    init_K
    sim
    change_K
    sim

-------------------
TEST 3:

    init_K
    change_K
    sim
    change_K
    sim


-------------------
TEST 4:

    init_K
    change_K
    sim
    change_K
    change_K
    sim

"""

import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../PlottingScripts/")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from plot_bcpnn_traces import TracePlotter
import pylab
import nest
import BCPNN

import numpy as np 

class TestBcpnn(object):

    def __init__(self, K_init):

        self.K_values = [K_init]
        # neuron 0 -> neuron 1
        tau_i = 10.
        tau_j = 10.
        tau_e = .1
        tau_p = 1000.
        fmax = 100.
#        epsilon = 1. / (fmax * tau_p)
        epsilon = 0.001
        bcpnn_init = 0.01
        p_i = 0.01#bcpnn_init
        p_j = 0.01#bcpnn_init
        weight = 0.
#        p_ij = np.exp(weight) * p_i * p_j
        p_ij = 0.0001 #np.exp(weight) * p_i * p_j
        self.syn_params  = {'p_i': p_i, 'p_j': p_j, 'p_ij': p_ij, 'gain': 1.0, 'K': K_init, \
                'fmax': fmax, 'epsilon': epsilon, 'delay':1.0, \
                'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}


    def init_sim(self):
        if 'bcpnn_synapse' not in nest.Models():
            try:
            #    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
                nest.Install('pt_module')
            except:
                nest.Install('pt_module')

        self.times_K_changed = [0.]
        nest.SetKernelStatus({'overwrite_files': True, \
                'resolution' : 0.1,
                })
        seed = 0
        nest.SetKernelStatus({'grng_seed' : 0})
        cell_params = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
            'E_in': -80.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -80.0, 'V_th': -60.0, \
            'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 5.0, 'tau_syn_in': 5.0}
        self.nrns = nest.Create('iaf_cond_exp_bias', 2, params=cell_params)
        self.voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.1})
        nest.SetStatus(self.voltmeter, [{"to_file": True, "withtime": True, 'label' : 'volt'}])
        nest.ConvergentConnect(self.voltmeter, self.nrns)
        self.exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'spikes'})
        nest.ConvergentConnect(self.nrns, self.exc_spike_recorder, delay=0.1, weight=1.)
        self.source_0 = nest.Create('spike_generator', 1)
        self.source_1 = nest.Create('spike_generator', 1)
        weight = 15.
        nest.Connect(self.source_0, [self.nrns[0]], weight, 1., model='static_synapse')
        nest.Connect(self.source_1, [self.nrns[1]], weight, 1., model='static_synapse')
        print 'syn_params init:', self.syn_params
        nest.Connect([self.nrns[0]], [self.nrns[1]], self.syn_params, model='bcpnn_synapse')

        debug = nest.GetStatus(nest.GetConnections([self.nrns[0]], [self.nrns[1]]))
        print 'debug syn status after initialization', debug
#        exit(1)

#        nest.SetStatus(nest.GetConnections([self.nrns[0]], [self.nrns[1]]), {'z_i': bcpnn_init, 'z_j': bcpnn_init, 'e_i': bcpnn_init, 'e_j': bcpnn_init})

#        nest.SetStatus(nest.GetConnections([self.nrns[0]], [self.nrns[1]]), {'z_i': 

        self.t_sim = 0.


    def set_spiking_behavior(self, pre_spikes=True, post_spikes=True, t_start=0., t_stop=100., dt=1., n_rnd_spikes=10):

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
        # post synaptic spikes 
        nest.SetStatus(self.source_0, {'spike_times' : spike_times_0})
        nest.SetStatus(self.source_1, {'spike_times' : spike_times_1})


    def get_nest_weight(self, debug=False):
        conns = nest.GetConnections([self.nrns[0]], [self.nrns[1]], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
        if len(conns) > 0:
            cp = nest.GetStatus([conns[0]])  # retrieve the dictionary for this connection
            pi = cp[0]['p_i']
            pj = cp[0]['p_j']
            pij = cp[0]['p_ij']
            if debug:
                print 'debug get_nest_weight:', cp[0]
            w = np.log(pij / (pi * pj))
        return w

    def trigger_pre_spike_and_get_nest_weight(self):
        spike_times_0 = [self.t_sim + .1]
        nest.SetStatus(self.source_0, {'spike_times' : spike_times_0})
        nest.Simulate(7.0)
        self.t_sim += 7.0

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
        return st_0, st_1

    def simulate(self, t_sim):
        nest.Simulate(t_sim)
        self.t_sim += t_sim

    def set_kappa(self, K, gain=0.):
        self.K_values.append(K)
        self.times_K_changed.append(self.t_sim)
        nest.SetStatus(nest.GetConnections([self.nrns[0]], [self.nrns[1]]), {'K': K, 'gain': gain, 't_k': nest.GetKernelStatus()['time']})


    def get_K_vec(self, dt=0.1):
        K_vec = self.K_values[0] * np.ones(self.t_sim / dt + 1) # + 10 because the spikewidth = 10
        if len(self.times_K_changed) > 1:
            for i_, t_change in enumerate(self.times_K_changed[1:]):
                idx = int(t_change / dt + 1)
                K_vec[idx:] = self.K_values[i_ + 1]
        return K_vec



def run_test_0(params, spike_pattern, K_values, plot=False):
    """
    TEST 0:

        init_K
        sim
    """

    nest.ResetKernel()
    np.random.seed(0)
    t_sim = params['t_sim']
    firing_rate = params['firing_rate']
    n_spikes = firing_rate * t_sim / 1000.
    Tester = TestBcpnn(K_values[0])
    ##### INIT K 
    Tester.init_sim()
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM 
    Tester.simulate(t_sim)
    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    test_results = {}
    test_results['w_nest'] = w_nest
    test_results['w_offline'] = w_offline
    test_results['w_diff_absolute'] = np.abs(w_nest - w_offline)
    test_results['w_diff_relative'] = w_diff_relative
    return test_results


def run_test_1(params, spike_pattern, K_values, plot=False):
    """
    TEST 1:

        init_K
        sim
        change_K
    """
    nest.ResetKernel()
    np.random.seed(0)
    t_sim = params['t_sim']
    n_spikes = params['firing_rate'] * t_sim / 1000.
    Tester = TestBcpnn(K_values[0])
    ##### INIT K 
    Tester.init_sim()
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM
    Tester.simulate(t_sim)
    ##### CHANGE K 
    Tester.set_kappa(K_values[1], gain=0.)
    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    test_results = {}
    test_results['w_nest'] = w_nest
    test_results['w_offline'] = w_offline
    test_results['w_diff_absolute'] = np.abs(w_nest - w_offline)
    test_results['w_diff_relative'] = w_diff_relative
    return test_results


def run_test_2(params, spike_pattern, K_values, plot=False):
    """
    TEST 2:

        init_K
        change_K 
        sim
    """
    nest.ResetKernel()
    np.random.seed(0)
    t_sim = params['t_sim']
    n_spikes = params['firing_rate'] * t_sim / 1000.
    Tester = TestBcpnn(K_values[0])
    ##### INIT K 
    Tester.init_sim()
    ##### CHANGE K 
    Tester.set_kappa(K_values[1], gain=0.)
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM 
    Tester.simulate(t_sim)
    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    test_results = {}
    test_results['w_nest'] = w_nest
    test_results['w_offline'] = w_offline
    test_results['w_diff_absolute'] = np.abs(w_nest - w_offline)
    test_results['w_diff_relative'] = w_diff_relative
    return test_results


def run_test_3A(params, spike_pattern, K_values, plot=False):
    """
    TEST 3 A:

        init_K
        sim
        change_K
        sim

    """
    debug_txt = 'DEEEEEEEEEEEEEEEEEEBUG  spike_pattern %s \tK_values %s' % (str(spike_pattern), str(K_values))
    print debug_txt

    nest.ResetKernel()
    np.random.seed(0)
    t_sim = params['t_sim']
    n_spikes = params['firing_rate'] * t_sim / 1000.
    Tester = TestBcpnn(K_values[0])
    ##### INIT K 
    Tester.init_sim()
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM
    Tester.simulate(t_sim)
    ##### CHANGE K AGAIN
    Tester.set_kappa(K_values[1], gain=0.)
    Tester.set_spiking_behavior(spike_pattern[2], spike_pattern[3], t_start = 10 + t_sim, t_stop= 2 * t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM
    Tester.simulate(t_sim)

    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    test_results = {}
    test_results['w_nest'] = w_nest
    test_results['w_offline'] = w_offline
    test_results['w_diff_absolute'] = np.abs(w_nest - w_offline)
    test_results['w_diff_relative'] = w_diff_relative
    return test_results



def run_test_3(params, spike_pattern, K_values, plot=False):
    """
    TEST 3:

        init_K
        change_K
        sim
        change_K
        sim

    """

    nest.ResetKernel()
    np.random.seed(0)
    t_sim = params['t_sim']
    n_spikes = params['firing_rate'] * t_sim / 1000.
    Tester = TestBcpnn(K_values[0])
    ##### INIT K 
    Tester.init_sim()
    ##### CHANGE K
    Tester.set_kappa(K_values[1], gain=0.)
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM
    Tester.simulate(t_sim)
    ##### CHANGE K AGAIN
    Tester.set_kappa(K_values[2], gain=0.)
    Tester.set_spiking_behavior(spike_pattern[2], spike_pattern[3], t_start = 10 + t_sim, t_stop= 2 * t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM
    Tester.simulate(t_sim)


    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()

    debug_txt = 'DEEEEEEEEEEEEEEEEEEBUG\nspike_pattern %s \tK_values %s\n' % (str(spike_pattern), str(K_values))
    debug_txt += 'spike trains: pre' + str(st_0) + '\npost:' + str(st_1)
    print debug_txt
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot)
    if plot:
        plot_voltages(Tester.voltmeter)
    w_diff = w_nest - w_offline
    

    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    print 'w_nest %.3e\tw_offline %.3e\trelative diff (to w_nest): %.2f' % (w_nest, w_offline, w_diff_relative)

    # build the test result dictionary
    test_results = {}
    test_results['w_nest'] = w_nest
    test_results['w_offline'] = w_offline
    test_results['w_diff_absolute'] = np.abs(w_nest - w_offline)
    test_results['w_diff_relative'] = w_diff_relative
    return test_results



def run_test_4(params, spike_pattern, K_values, plot=False):
    """
    TEST 4:

        init_K
        change_K
        sim
        change_K
        change_K
        sim
    """

    nest.ResetKernel()
    np.random.seed(0)
    t_sim = params['t_sim']
    n_spikes = params['firing_rate'] * t_sim / 1000.
    Tester = TestBcpnn(K_values[0])
    ##### INIT K 
    Tester.init_sim()
    ##### CHANGE K
    Tester.set_kappa(K_values[1], gain=0.)
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    ##### SIM
    Tester.simulate(t_sim)
    ##### CHANGE K AGAIN
    Tester.set_kappa(K_values[2], gain=0.)
    Tester.set_kappa(K_values[3], gain=0.)
    Tester.simulate(t_sim)
    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    test_results = {}
    test_results['w_nest'] = w_nest
    test_results['w_offline'] = w_offline
    test_results['w_diff_absolute'] = np.abs(w_nest - w_offline)
    test_results['w_diff_relative'] = w_diff_relative
    return test_results




def compute_bcpnn_traces(spike_train_0, spike_train_1, K_vec, syn_params, t_sim, plot=False, save=False):
    #######################
    # OFFLINE COMPUTATION
    #######################
    s_pre = BCPNN.convert_spiketrain_to_trace(spike_train_0, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(spike_train_1, t_sim)
    bcpnn_traces = []
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, syn_params, K_vec=K_vec)
    w_end = wij[-1]
    bcpnn_traces = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    if plot:
        output_fn = 'offline_traces.png'
        dt = 0.1
        TP = TracePlotter({})
#        TP.plot_trace(bcpnn_traces, bcpnn_params, dt, output_fn=None, info_txt=None, fig=None, \
#                color_pre='b', color_post='g', color_joint='r', style_joint='-')
        TP.plot_trace_with_spikes(bcpnn_traces, syn_params, dt, output_fn=output_fn, fig=None, \
                color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=K_vec, \
                extra_txt='K_vec vs Time')
        print 'Saving trace figure to:', output_fn
    if save:
        output_fn = 'offline_traces.dat'
        np.savetxt(output_fn, np.array(bcpnn_traces).transpose())
        print 'Saving trace data to:', output_fn
    return w_end



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


if __name__ == '__main__':

    # run a simple test

    # for test_3
    spike_pattern = (True, True, False, True)
    K_values = (0., 1.)

    # for test 3A
#    spike_pattern = (False, True, False, False)
#    K_values = (1.0, 1.0)

    # for Test 4
    spike_pattern = (False, True, True, True)
    K_values = (0.0, 1.0, 1.0, 1.0)

    nest.ResetKernel()
    np.random.seed(0)
    t_sim = 1000.
    firing_rate = 50.
    n_spikes = firing_rate * t_sim / 1000.
    Tester = TestBcpnn(K_values[0])

    ##### INIT K 
    Tester.init_sim()

    w_nest_init = Tester.get_nest_weight(debug=True)

    # for test_3
    ##### UPDATE K 
#    Tester.set_kappa(K=K_values[1], gain=1.)
    # for test 3A: no update K here

    ##### SIM 
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    Tester.simulate(t_sim)

    ##### UPDATE K
    # for Test 3A
#    Tester.set_kappa(K=K_values[1], gain=1.)
    # for Test 3
#    Tester.set_kappa(K=K_values[2], gain=1.)
    
    # for Test 4
    Tester.set_kappa(K=K_values[2], gain=1.)
    Tester.set_kappa(K=K_values[3], gain=1.)


    ##### SIM 
    Tester.set_spiking_behavior(spike_pattern[2], spike_pattern[3], t_start=10 + t_sim, t_stop=2 * t_sim-20., n_rnd_spikes=n_spikes)
    Tester.simulate(t_sim)

    print 'w_nest_init:', w_nest_init
    print 'times_K_changed :', Tester.times_K_changed
    print 'K_values:', Tester.K_values
    print 'DEBUG t_sim before trigger_pre_spike_and_get_nest_weight', Tester.t_sim
    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    print 'DEBUG t_sim after trigger_pre_spike_and_get_nest_weight', Tester.t_sim
    st_0, st_1 = Tester.get_spike_trains()
    print 'DEBUG123 spiketrain pre :', st_0
    print 'DEBUG123 spiketrain post:', st_1
    K_vec = Tester.get_K_vec()
    w_offline = compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot=True, save=True)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    
    print 'w_nest:', w_nest
    print 'w_offline:', w_offline
    print 'w_diff_relative:', w_diff_relative
    print 'w_diff_absolute:', np.abs(w_nest - w_offline)
#    plot_voltages(Tester.voltmeter)
    pylab.show()
