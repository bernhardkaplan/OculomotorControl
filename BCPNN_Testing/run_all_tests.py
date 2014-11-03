import numpy as np 
import pylab
import itertools
import test_functions


def run_all_test_of_one_type(callback, n_sim, n_k, params):

    test_txt = ''
    test_txt += 'Function: %s\n' % (callback.__name__)
    test_txt += 'wdiff abs\twdiff rel\tw_nest\tw_offline\tspike_pattern\t K_pattern\n'

    # re-initialize the seed
    np.random.seed(0)

    K_combinations = itertools.product([0., 1.], repeat=n_k) # K must be a float
    for K_pattern in K_combinations:
        print 'DEBUG K_pattern:', K_pattern
        # number of simulations determines the number of possible pre/post spike combinations
        # itertools products need to be recreated after iterating over them!
        spike_combinations = itertools.product([True, False], repeat=2 * n_sim)
        for spike_pattern in spike_combinations:
            test_results = callback(params, spike_pattern, K_pattern)
            test_txt += '%.2e\t%.2f\t%.3f\t%.3f\t%s\t%s\n' % (test_results['w_diff_absolute'], test_results['w_diff_relative'], test_results['w_nest'], test_results['w_offline'], str(spike_pattern), str(K_pattern))

    return test_txt


if __name__ == '__main__':


    params = {}
    params['t_sim'] = 1000.
    params['firing_rate'] = 50.
    # single debug run for test 3
#    spike_pattern = [False, True, True, False]
#    K_pattern = [0., 1., 1.]

#    spike_pattern = [False, True, True, True]
#    K_pattern = [0., 0., 0.]

#    spike_pattern = (False, False, True, True)
#    K_pattern = (0., 0., 0.)

#    spike_pattern = (True, False, True, True)
#    K_pattern = (0., 1., 0.)
#    test_results = ''
#    test_results += 'wdiff\t spike_pattern\t K_pattern\n'
#    w_diff_rel = test_functions.run_test_3(spike_pattern, K_pattern, plot=True)
#    test_results += '%.2f\t%s\t%s\n' % (w_diff_rel, str(spike_pattern), str(K_pattern))
#    print 'test results', test_results
#    pylab.show() 

#    result_0 = run_all_test_of_one_type(test_functions.run_test_0, 1, 1, params)
#    result_1 = run_all_test_of_one_type(test_functions.run_test_1, 1, 2, params)
#    result_2 = run_all_test_of_one_type(test_functions.run_test_2, 1, 2, params)
#    result_3 = run_all_test_of_one_type(test_functions.run_test_3A, 2, 2, params)
#    result_3 = run_all_test_of_one_type(test_functions.run_test_3, 2, 3, params)
    result_4 = run_all_test_of_one_type(test_functions.run_test_4, 2, 4, params)
#    print 'Result 0:', result_0
#    print 'Result 1:', result_1
#    print 'Result 2:', result_2
#    print 'Result 3:', result_3
    print 'Result 4:', result_4


#    w_diff_rel = test_functions.run_test_0(pre_spikes=True, post_spikes=False, K=1.)

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
