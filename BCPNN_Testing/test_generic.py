import os, sys, inspect
import nest
import numpy as np 
import pylab
#from test_functions import TestBcpnn, compute_bcpnn_traces, plot_voltages
import test_functions as TF


if __name__ == '__main__':

    # run a simple test
    spike_pattern = [True, False, True, False]
    K_values = [1., 0., 1.]
#    run_test_0(spike_pattern, K_values)

    nest.ResetKernel()
    np.random.seed(0)
    t_sim = 1000.
    firing_rate = 1.
    n_spikes = firing_rate * t_sim / 1000.
    Tester = TF.TestBcpnn()
    ##### INIT K 
    Tester.init_sim(K=K_values[0])

    ##### UPDATE K
#    Tester.set_kappa(K=K_values[1], gain=0.)

    ##### SIM 
    Tester.set_spiking_behavior(spike_pattern[0], spike_pattern[1], t_start=10, t_stop=t_sim-10., n_rnd_spikes=n_spikes)
    Tester.simulate(t_sim)

    ##### UPDATE K
#    Tester.set_kappa(K=K_values[2], gain=0.)

    ##### SIM 
    Tester.set_spiking_behavior(spike_pattern[2], spike_pattern[3], t_start=10 + t_sim, t_stop=2 * t_sim-10., n_rnd_spikes=n_spikes)
    Tester.simulate(t_sim)

    w_nest = Tester.trigger_pre_spike_and_get_nest_weight()
    st_0, st_1 = Tester.get_spike_trains()
    print 'Pre synaptic spikes:', st_0
    print 'Post synaptic spikes:', st_1
    K_vec = Tester.get_K_vec()

    w_offline = TF.compute_bcpnn_traces(st_0, st_1, K_vec, Tester.syn_params, Tester.t_sim, plot=True)
    w_diff = w_nest - w_offline
    if w_nest != 0.:
        w_diff_relative = w_diff / w_nest * 100.
    elif w_nest == 0 and w_offline == 0.:
        w_diff_relative = 0.
    elif w_nest == 0 and w_offline != 0.:
        w_diff_relative = np.inf
    
    print 'w_nest:', w_nest
    print 'w_offline:', w_offline
    print 'w_diff_relative', w_diff_relative
    TF.plot_voltages(Tester.voltmeter)
    pylab.show()
