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
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)

    TP.plot_trace(bcpnn_traces_0, bcpnn_params, dt, output_fn=output_fn, info_txt='Stim 0 - Action\n', fig=fig, \
            color_pre='b', color_post='k', color_joint='b', style_joint='--')
    TP.plot_trace(bcpnn_traces_1, bcpnn_params, dt, output_fn=output_fn, info_txt='Stim 1 - Action\n', fig=fig, \
            color_pre='g', color_post='m', color_joint='g', style_joint='--')


def run_Xstim_1action(bcpnn_params, n_stim):
    t_iteration = 50
    t_sim = 4 * t_iteration

    firing_rate = 100
    pre_traces = []
    for i_stim in xrange(n_stim):
        spikes = create_spikes(i_stim * t_iteration, firing_rate)
        pre_traces.append(BCPNN.convert_spiketrain_to_trace(spikes, t_sim))

    action_0 = create_spikes(0, t_iteration, firing_rate)
    s_post = BCPNN.convert_spiketrain_to_trace(action_0, t_sim)


if __name__ == '__main__':

#    TP.plot_trace(bcpnn_traces[1], bcpnn_params, dt, output_fn=output_fn, info_txt=info_txt)
    np.random.seed(0)
    tau_i = 5.
    tau_j = 5.
    tau_e = 5.
    tau_p = 50000.
    bcpnn_init = 0.01
    gain = 0.
    K = 1.
    fmax = 150.
    epsilon = 1. / (fmax * tau_p)
    bcpnn_params = {'p_i': bcpnn_init, 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': gain, 'K': K, \
            'fmax': fmax, 'epsilon': epsilon, 'delay': 1.0, \
            'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}

    run_2stim_1action(bcpnn_params)
    run_Xstim_1action(bcpnn_params, n_stim)

    pylab.show()
