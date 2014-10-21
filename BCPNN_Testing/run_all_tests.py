import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../PlottingScripts/")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from plot_bcpnn_traces import TracePlotter
import numpy as np 
import pylab
import itertools
import test_functions

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




#def run_all_test_of_one_type(callback):
#    test_data = []
#    for K in [0., 1.]:
#        spike_combinations = itertools.product([True, False], repeat=2)
#        for pre_spikes, post_spikes in spike_combinations:
#            w_diff_rel = callback(pre_spikes, post_spikes, K=K)
#            test_data.append((w_diff_rel, pre_spikes, post_spikes, K))
#    spike_combinations = itertools.product([True, False], repeat=2)
#    test_results = ''
#    test_results += 'Function: %s\n' % (callback.__name__)
#    for d in test_data:
#        test_results += '%.2f\t\t%s\t%s\t%.1f\n' % (d[0], d[1], d[2], d[3])
#    return test_results


def run_all_test_of_one_type(callback, n_sim, n_k):
    test_data = []


    test_results = ''
    test_results += 'Function: %s\n' % (callback.__name__)
    test_results += 'wdiff\t spike_pattern\t K_pattern\n'

    K_combinations = itertools.product([0., 1.], repeat=n_k)
    for K_pattern in K_combinations:
        print 'DEBUG K_pattern:', K_pattern
        # number of simulations determines the number of possible pre/post spike combinations
        # itertools products need to be recreated after iterating over them!
        spike_combinations = itertools.product([True, False], repeat=2 * n_sim)
        for spike_pattern in spike_combinations:
            w_diff_rel = callback(spike_pattern, K_pattern)
            test_results += '%.2f\t%s\t%s\n' % (w_diff_rel, str(spike_pattern), str(K_pattern))
    return test_results


if __name__ == '__main__':

#    result_0 = run_all_test_of_one_type(test_functions.run_test_0, 1, 1)
#    result_1 = run_all_test_of_one_type(test_functions.run_test_1, 1, 2)
#    result_2 = run_all_test_of_one_type(test_functions.run_test_2, 1, 2)
    result_3 = run_all_test_of_one_type(test_functions.run_test_3, 2, 3)
#    result_4 = run_all_test_of_one_type(test_functions.run_test_4, 2, 2)

#    print 'Result 0:', result_0
#    print 'Result 1:', result_1
#    print 'Result 2:', result_2
    print 'Result 3:', result_3
#    print 'Result 4:', result_4


#    w_diff_rel = test_functions.run_test_0(pre_spikes=True, post_spikes=False, K=1.)
#    pylab.show() 

    # TODO: create different test cases and for each test case
    # run the varies options:
    #  init_K (K=0) (K=1)
    #  change_K (K --> K) K remains the same
    #  change_K (K --> not K) K is set to the opposite (from 0 --> 1 or the other way around)
    #  change_K (K --> K + .5) change to a different value 
    #  sim with / without pre- and/or post firing


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


TEST 5:
    init_K
    change_K
    change_K
    sim
    change_K
    sim
"""
