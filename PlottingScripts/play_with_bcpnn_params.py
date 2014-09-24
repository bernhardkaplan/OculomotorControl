import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import json
import numpy as np
import utils
from PlottingScripts.plot_bcpnn_traces import TracePlotter
from PlottingScripts.plot_voltages import VoltPlotter
import pylab
from PlottingScripts import FigureCreator
import BCPNN

FigureCreator.plot_params['figure.subplot.left'] = .17
pylab.rcParams.update(FigureCreator.plot_params)

import simulation_parameters


def create_spikes(t_start, t_stop, rate):
    assert (t_stop > t_start)
    n_spikes = (t_stop - t_start) / 1000. * rate
    spikes = np.random.uniform(t_start, t_stop, n_spikes)
    return spikes



def run_2stim_1action(bcpnn_params):
    t_iteration = 50
    t_sim = 4 * t_iteration
    firing_rate = 100

    stim_0 = create_spikes(0, t_iteration, firing_rate)
    stim_1 = create_spikes(t_iteration, 2 * t_iteration, firing_rate)
    action_0 = create_spikes(0, t_iteration, firing_rate)

    s_pre = BCPNN.convert_spiketrain_to_trace(stim_0, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(action_0, t_sim)
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, bcpnn_params)
    bcpnn_traces_0 = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]
    print 'debug', len(bcpnn_traces_0)

    s_pre = BCPNN.convert_spiketrain_to_trace(stim_1, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(action_0, t_sim)

    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, bcpnn_params)
    bcpnn_traces_1 = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    params = None
    TP = TracePlotter(params)
    output_fn = None
    info_txt = None
    dt = .1

    fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
    TP.plot_trace(bcpnn_traces_0, bcpnn_params, dt, output_fn=output_fn, info_txt='Stim 0 - Action\n', fig=fig, \
            color_pre='b', color_post='k', color_joint='b', style_joint='--')
    TP.plot_trace(bcpnn_traces_1, bcpnn_params, dt, output_fn=output_fn, info_txt='Stim 1 - Action\n', fig=fig, \
            color_pre='g', color_post='m', color_joint='g', style_joint='--')


def run_Xstim_1action(bcpnn_params, n_stim):
    t_active = 25
    t_inactive = 175
    t_iteration = t_active + t_inactive

    t_sim = n_stim * t_iteration

    firing_rate = 250
    pre_spikes = np.array([])
    for i_stim in xrange(n_stim):
        # create pre synaptic 
        spikes = create_spikes(i_stim * t_iteration, i_stim * t_iteration + t_active, firing_rate)
        pre_spikes = np.r_[pre_spikes, spikes]

    # no post activity, inactivity
#    post_spikes = np.array([])

    # high post activity
    post_spikes = create_spikes(0, i_stim * t_iteration + t_active, firing_rate)

    s_pre = BCPNN.convert_spiketrain_to_trace(pre_spikes, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(post_spikes, t_sim)
    K_vec = np.ones(s_pre.size)
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, bcpnn_params, K_vec=K_vec, w_init=.0)
    bcpnn_traces_pre = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

#    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, bcpnn_params)
#    bcpnn_traces_post = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    params = None
    TP = TracePlotter(params)
    output_fn = None
    info_txt = None
    dt = .1

    fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
    TP.plot_trace(bcpnn_traces_pre, bcpnn_params, dt, output_fn=output_fn, info_txt='Stim 0 - Action\n', fig=fig, \
            color_pre='b', color_post='k', color_joint='b', style_joint='--')

#    TP.plot_trace(bcpnn_traces_post, bcpnn_params, dt, output_fn=output_fn, info_txt='Stim 1 - Action\n', fig=fig, \
#            color_pre='g', color_post='m', color_joint='g', style_joint='--')


def plot_bcpnn_traces(bcpnn_params, pre_spike_train, post_spike_train):

#    t_sim = max(np.max(pre_spike_train), np.max(post_spike_train))
    t_sim = 306.2
    s_pre = BCPNN.convert_spiketrain_to_trace(pre_spike_train, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(post_spike_train, t_sim)
    K_vec = .0 * np.ones(s_pre.size)

    K_vec[1500:2000] = .7
    K_vec[2000:2500] = .3
    K_vec[2500:] = .0
    print 'K_vec:', K_vec

    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, bcpnn_params, K_vec=K_vec, w_init=.0)
    bcpnn_traces_pre = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

#    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, bcpnn_params)
#    bcpnn_traces_post = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    params = None
    TP = TracePlotter(params)
    output_fn = None
    info_txt = None
    dt = .1

    print 'Debug bcpnn_params:', bcpnn_params
    print 'Final weight:', wij[-1]
#    fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
    TP.plot_trace(bcpnn_traces_pre, bcpnn_params, dt, output_fn=output_fn, 
            color_pre='b', color_post='k', color_joint='b', style_joint='--')



if __name__ == '__main__':

#    TP.plot_trace(bcpnn_traces[1], bcpnn_params, dt, output_fn=output_fn, info_txt=info_txt)
    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    cell_type_post = 'd1'
    bcpnn_params = params['params_synapse_%s_MT_BG' % cell_type_post]
    bcpnn_params['tau_j'] = 5.
    bcpnn_params['tau_i'] = 5.
    bcpnn_params['tau_e'] = 300.
    bcpnn_params['tau_p'] = 50000.
    bcpnn_params['gain'] = 1.
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
    np.random.seed(0)

    pre_spikes = np.array([56.4, 106.4, 306.2])
    post_spikes = np.array([57.4, 107.4, 207.2, 228.4, 242.4, 305.9])

    plot_bcpnn_traces(bcpnn_params, pre_spikes, post_spikes)
#    run_2stim_1action(bcpnn_params)
#    n_stim = 10
#    run_Xstim_1action(bcpnn_params, n_stim)

    pylab.show()
