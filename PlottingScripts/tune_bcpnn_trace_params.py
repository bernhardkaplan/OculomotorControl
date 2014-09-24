"""
This script is to plot spikes and BCPNN traces from simulated spike trains 
and gives the opportunity to play with the BCPNN parameters without re-simulating the whole network.
"""

import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import sys
import os
import utils
import re
import numpy as np
import pylab
import simulation_parameters
import json
import itertools
import BCPNN
from plot_bcpnn_traces import TracePlotter
from PlottingScripts.plot_reward_learning_traces import create_K_vector

#import MergeSpikefiles
#import PlotMPNActivity
#import FigureCreator



def select_most_active_cells(params, spike_data, it_range):

    t_range = np.array(it_range) * params['t_iteration']
    time_filtered_spikes = utils.get_spiketimes_within_interval(spike_data, t_range[0], t_range[1])
    (pre_gids, nspikes) = utils.get_most_active_neurons(time_filtered_spikes)
    return pre_gids



def get_spikes_for_gids(spike_data, gids):

    spike_times = {gid : None for gid in gids}
    for gid in gids:
        idx = (spike_data[:, 0] == gid).nonzero()[0]
        spike_times[gid] = spike_data[idx, 1]
    return spike_times


if __name__ == '__main__':

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    
    # load spikes
    cell_type_post = 'd2'
    bcpnn_params = params['params_synapse_%s_MT_BG' % cell_type_post]
    fn_spikes_pre = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    fn_spikes_post = params['spiketimes_folder'] + params['%s_spikes_fn_merged_all' % cell_type_post]
    if (not os.path.exists(fn_spikes_pre)) or (not os.path.exists(fn_spikes_post)):
        utils.merge_spikes(params)
    all_spikes_pre = np.loadtxt(fn_spikes_pre)
    all_spikes_post = np.loadtxt(fn_spikes_post)

    # load BG cell gids
    f = file(params['bg_gids_fn'], 'r')
    bg_gids = json.load(f) 
    action_idx = 1
    post_gids = bg_gids[cell_type_post][action_idx]
    print 'Plotting post gids:', post_gids
    n_post = len(post_gids)

    # PARAMETERS
    it_range_pre_selection = (0, 1) # sets time frame for pre synaptic cell selection
    it_range_bcpnn_computations = (0, 1 * params['n_iterations'])
    t_range_bcpnn_computations = np.array(it_range_bcpnn_computations) * params['t_iteration']
    dt = params['dt']

    bcpnn_params['fmax'] = 100.
    try:
        bcpnn_params['tau_p'] = float(sys.argv[5])
        bcpnn_params['tau_e'] = float(sys.argv[4])
        bcpnn_params['tau_j'] = float(sys.argv[3])
        bcpnn_params['tau_i'] = float(sys.argv[2])
        bcpnn_params['gain'] = 1.
        show = False
    except:
        print '\n\tTaking BCPNN parameters from simulation_parameters!\n'
        show = True

#    bcpnn_params['tau_p'] = 2400 * 10
#    bcpnn_params['tau_i'] = 50 # 50 
#    bcpnn_params['tau_j'] = 10 # 3

    # select pre-synaptic cells
    pre_gids = select_most_active_cells(params, all_spikes_pre, it_range_pre_selection)
    pre_gids = list(pre_gids)
    pre_gids.reverse()
    print 'Plotting pre gids:', pre_gids

    
    # filter all_spikes for gids
    spike_times_pre = get_spikes_for_gids(all_spikes_pre, pre_gids)
    spike_times_post = get_spikes_for_gids(all_spikes_post, post_gids)
#    print 'Spike times pre:', spike_times_pre
#    print 'Spike times post:', spike_times_post

    # for all pairs of pre- and post-cells, compute traces
    gid_pairs = list(itertools.product(pre_gids, post_gids))
    print 'number of gid_pairs:', len(gid_pairs)

    n_traces_to_compute = 5 # for debugging and development
    bcpnn_traces = []


    # create a vector controlling the weight update 
    K_vec = create_K_vector(params, it_range_bcpnn_computations, dt=dt)
    np.savetxt('delme_kvec', K_vec)

    # compute 
    fig = None
    for i_trace in xrange(n_traces_to_compute):
        pre_gid = gid_pairs[i_trace][0]
        post_gid = gid_pairs[i_trace][1]

        time_filtered_spikes_pre = utils.get_spiketimes_within_interval(spike_times_pre[pre_gid], t_range_bcpnn_computations[0], t_range_bcpnn_computations[1])
        time_filtered_spikes_post = utils.get_spiketimes_within_interval(spike_times_post[post_gid], t_range_bcpnn_computations[0], t_range_bcpnn_computations[1])
        s_pre = BCPNN.convert_spiketrain_to_trace(time_filtered_spikes_pre , t_range_bcpnn_computations[1])
        s_post = BCPNN.convert_spiketrain_to_trace(time_filtered_spikes_post , t_range_bcpnn_computations[1])

        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, params['params_synapse_%s_MT_BG' % cell_type_post], \
                K_vec=K_vec, w_init=1.)
        bcpnn_traces.append([wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post])

    # plotting 
    TP = TracePlotter(params, cell_type_post)
    fig = None
    output_fn = None 
    for i_, traces in enumerate(bcpnn_traces):
#        output_fn = output_fn_base + '%d_%d.png' % (gid_pairs[i_][0], gid_pairs[i_][1])
        fig = TP.plot_trace_with_spikes(bcpnn_traces[i_], bcpnn_params, dt, output_fn=output_fn, fig=fig, K_vec=K_vec)

    ax_rp = fig.axes[1]
#    plot_

    output_fn = params['figures_folder'] + 'bcpnn_traces_RBL_%s_action%d_tau_zi%04d_zj%04d_e%04d_p%04d_gain%.1f.png' % (cell_type_post, action_idx, bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'], bcpnn_params['gain'])
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)

    if show:
        pylab.show()


#    bcpnn_params['K'] = 1.
#    dt = params['dt']
