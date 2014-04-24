import os
import sys
import json
import numpy as np
import utils
from PlottingScripts.plot_bcpnn_traces import TracePlotter
import pylab


def check_spike_files(params, cell_type_post='d1'): 
    fn_pre = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    fn_post = params['spiketimes_folder'] + params['%s_spikes_fn_merged_all' % cell_type_post]
    if (not os.path.exists(fn_pre)) or (not os.path.exists(fn_post)):
        utils.merge_spikes(params)
    print 'Debug, spike data exists:', os.path.exists(fn_pre), os.path.exists(fn_post)
    print 'Debug, spike data sizes:', os.path.getsize(fn_pre), os.path.getsize(fn_post)


def get_weights(pre_gids, post_gids, tgt_cell_type='d1'):
    conn_list_fn = training_params['mpn_bg%s_merged_conn_fn' % tgt_cell_type]
    print 'Loading weights from:', conn_list_fn
    if not os.path.exists(conn_list_fn):
        print 'Merging default connection files...'
        merge_pattern = params['mpn_bgd1_conn_fn_base']
        fn_out = params['mpn_bgd1_merged_conn_fn']
        utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)


if __name__ == '__main__':

    training_params = utils.load_params( os.path.abspath(sys.argv[1]) )
    testing_params = utils.load_params(os.path.abspath(sys.argv[2]))

    it_range_global = (0, 3)
    n_pre = 5
    n_post = 5
    cell_type_post = 'd1'

    check_spike_files(training_params)
    check_spike_files(testing_params)

    TP_training = TracePlotter(training_params)
    fn_pre = training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged']
    fn_post = training_params['spiketimes_folder'] + training_params['%s_spikes_fn_merged_all' % cell_type_post]
    TP_training.load_spikes(fn_pre, fn_post)

    all_pre_gids = []
    all_post_gids = []

    all_gid_pairs = {}
    trace_buffer = {}


    for it in xrange(it_range_global[0], it_range_global[1]):
        it_range = (it, it+1)
        pre_gids, post_gids = TP_training.select_cells(n_pre=n_pre, n_post=n_post, it_range=it_range)
        all_pre_gids += list(pre_gids)
        all_post_gids += list(post_gids)

        pre_traces, gid_pairs = TP_training.compute_traces(pre_gids, post_gids)
        all_gid_pairs[it] = gid_pairs

        for i_, (pre_gid, post_gid) in enumerate(gid_pairs):
            trace_buffer[(pre_gid, post_gid)] = {}
            trace_buffer[(pre_gid, post_gid)][it] = {}
            trace_buffer[(pre_gid, post_gid)][it]['wij'] = pre_traces[i_][0]
            trace_buffer[(pre_gid, post_gid)][it]['pij'] = pre_traces[i_][4]
            trace_buffer[(pre_gid, post_gid)][it]['eij'] = pre_traces[i_][7]

            trace_buffer[pre_gid] = {}
            trace_buffer[pre_gid][it] = {}
            trace_buffer[pre_gid][it]['pi'] = pre_traces[i_][2]
            trace_buffer[pre_gid][it]['ei'] = pre_traces[i_][5]
            trace_buffer[pre_gid][it]['zi'] = pre_traces[i_][8]

            trace_buffer[post_gid] = {}
            trace_buffer[post_gid][it] = {}
            trace_buffer[post_gid][it]['bias'] = pre_traces[i_][1]
            trace_buffer[post_gid][it]['pj'] = pre_traces[i_][3]
            trace_buffer[post_gid][it]['ej'] = pre_traces[i_][6]
            trace_buffer[post_gid][it]['zj'] = pre_traces[i_][9]

#                  0    1      2   3    4   5   6   7   8   9   10      11
#                ([wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post])
#    print 'You should (re-)run the testing with:'
#    print 'gids_to_record_mpn:', all_pre_gids
#    print 'gids_to_record_bg:', all_post_gids

#    for it in xrange(it_range_global[0], it_range_global[1]):
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    it = 0
    for i_, (pre_gid, post_gid) in enumerate(all_gid_pairs[it]):
        print 'pre_gid, post', pre_gid, post_gid
        wij = trace_buffer[(pre_gid, post_gid)][it]['wij']
        x = np.arange(wij.size)
        ax.plot(x, wij, label='%d-%d' % (pre_gid, post_gid))
    ax.set_title('it = %d' % it)

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    it = 1
    for i_, (pre_gid, post_gid) in enumerate(all_gid_pairs[it]):
        print 'pre_gid, post', pre_gid, post_gid
        wij = trace_buffer[(pre_gid, post_gid)][it]['wij']
        x = np.arange(wij.size)
        ax.plot(x, wij, label='%d-%d' % (pre_gid, post_gid))

    # plot voltages
    ax.set_title('it = %d' % it)
    pylab.show()
