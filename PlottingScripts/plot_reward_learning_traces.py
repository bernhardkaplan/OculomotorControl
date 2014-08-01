import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import json
import numpy as np
import utils
from plot_bcpnn_traces import TracePlotter
from plot_voltages import VoltPlotter
import pylab
import FigureCreator
import simulation_parameters

#FigureCreator.plot_params['figure.subplot.left'] = .17
#pylab.rcParams.update(FigureCreator.plot_params)


def create_K_vector(params, it_range, dt=0.1):

    rewards = np.loadtxt(params['rewards_given_fn']) 
    print 'K_vec', rewards
    t_max = it_range[1] * params['t_iteration']
    n = np.int(t_max/ dt) + 1 # +1 because the length needs to be the same computed in the BCPNN module after convert_spiketrain_to_trace
    K_vec = np.zeros(n)
    for it_ in xrange(it_range[0], it_range[1]):
        idx_1 = np.int(it_ * params['t_iteration'])
        idx_2 = np.int((it_ + 1) * params['t_iteration'])
        K_vec[idx_1:idx_2] = rewards[it_]
    return K_vec            


if __name__ == '__main__':


    n_pre = 2
    n_post = 1
    it_range_selection = (0, 2)
    it_range_plotting = (0, 5)
    output_fn = None 
    info_txt = None

    # get params
    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    # merge spike files if needed
    cell_type_post = 'd1'
    dt = params['dt']
    fn_pre = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    fn_post = params['spiketimes_folder'] + params['%s_spikes_fn_merged_all' % cell_type_post]
    if (not os.path.exists(fn_pre)) or (not os.path.exists(fn_post)):
        utils.merge_spikes(params)

    bcpnn_params = params['params_synapse_%s_MT_BG' % cell_type_post]

    K_vec = create_K_vector(params, it_range_plotting, dt=0.1)
    TP = TracePlotter(params, cell_type_post)
    TP.load_spikes(fn_pre, fn_post)
    pre_gids, post_gids = TP.select_cells(n_pre=n_pre, n_post=n_post, it_range=it_range_selection)
    print 'pre_gids', pre_gids
    print 'post_gids', post_gids
    all_traces, gid_pairs = TP.compute_traces(pre_gids, post_gids, it_range_plotting, K_vec=K_vec)

    fig = None
    for i_, traces in enumerate(all_traces):
#        output_fn = output_fn_base + '%d_%d.png' % (gid_pairs[i_][0], gid_pairs[i_][1])
        info_txt = 'Pre: %d  Post: %d' % (gid_pairs[i_][0], gid_pairs[i_][1])
        fig = TP.plot_trace(traces, bcpnn_params, dt, output_fn=output_fn, info_txt=info_txt, fig=fig)
    pylab.show()
    
#     load spike files 
#    print 'Loading spike data from:', fn_pre, '\n', fn_post
#    pre_spikes = np.loadtxt(fn_pre)
#    post_spikes = np.loadtxt(fn_post)

#     filter spikes to given time range and extract spike trains for active cells
#        self.d_pre = utils.get_spiketimes_within_interval(self.pre_spikes, self.t_range[0], self.t_range[1])
#        self.d_post = utils.get_spiketimes_within_interval(self.post_spikes, self.t_range[0], self.t_range[1])

    # get bcpnn traces from spike trains

    # plot pre and post synaptic traces
