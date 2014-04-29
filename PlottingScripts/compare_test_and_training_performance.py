import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import matplotlib
import sys
import os
import utils
import json
from FigureCreator import plot_params
pylab.rcParams.update(plot_params)
from PlotMPNActivity import ActivityPlotter

class Plotter(object):

    def __init__(self):
        self.plots = []
        self.legend_labels = []

    def set_training_params(self, training_params):
        self.training_params = training_params

    def get_xdisplacement(self, params, stim_range):

        Plotter = ActivityPlotter(params)#, it_max=1)
        (t_axis, x_displacement) = Plotter.plot_retinal_displacement(stim_range=stim_range)
        return (t_axis, x_displacement)
    
    def plot_xdisplacement(self, params, color='k', ls='-', ax=None, legend_label='', plot_vertical_lines=True, stim_range=None):

        if ax == None:
            fig = pylab.figure()
            ax = fig.add_subplot(111)
        ax.set_title('Comparison of training vs. test performance')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Retinal displacement')
        fn = params['data_folder'] + 'mpn_xdisplacement.dat'
        if not os.path.exists(fn):
            self.get_xdisplacement(params, stim_range)

        print 'Loading data:', fn
        d = np.loadtxt(fn)

        t_stim = params['t_iteration'] * params['n_iterations_per_stim']
        xdispl = np.abs(d[:, 0] - .5)
        t_axis = d[:, 4]
        t_axis += .5 * params['t_iteration']
        n_iterations = d[:, 0].size
        if stim_range == None:
            stim_range = (0, params['n_stim'])
        if params['training']:
            t_offset = (-1.) * t_stim * stim_range[0]
            # you have a large arrary of xdispl 
            for stim in xrange(stim_range[0], stim_range[-1] + 1):
                for it_ in xrange(params['n_iterations_per_stim'] - 2):
                    it_1 = it_ + stim * params['n_iterations_per_stim']
                    it_2 = it_ + stim * params['n_iterations_per_stim'] + 2
                    p1, = ax.plot(t_axis[it_1:it_2] + t_offset, xdispl[it_1:it_2], c=color, lw=3, ls=ls)
        else:
            # different indexing of training stimuli
            t_offset = 0
            for stim, stim_idx in enumerate(stim_range): # stim_idx is not used
                for it_ in xrange(params['n_iterations_per_stim'] - 2):
                    it_1 = it_ + stim * params['n_iterations_per_stim']
                    it_2 = it_ + stim * params['n_iterations_per_stim'] + 2
                    p1, = ax.plot(t_axis[it_1:it_2], xdispl[it_1:it_2], c=color, lw=3, ls=ls)
        print 't_axis[it_1:it_2], xdispl[it_1:it_2]', t_axis[it_1:it_2], xdispl[it_1:it_2]
        print 'debug', params['n_stim'], stim_range

#        p1, = ax.plot(t_axis, xdispl, c=color, lw=3, ls=ls)
        self.plots.append(p1)
        self.legend_labels.append(legend_label)

        if plot_vertical_lines:
            AP = ActivityPlotter(params)
            AP.plot_vertical_lines(ax)

        time_range = (stim_range[0] * t_stim + t_offset, (stim_range[-1] + 1) * t_stim + t_offset)
        ax.set_xlim(time_range)
        return ax


#    def compare_xdisplacement(self):
#        p2, = ax.plot(t_axis, test_xdispl, c='b', ls='--', lw=3)
#        plots = [p1, p2]
#        ax.legend(plots, ['training', 'test'], loc='upper left')


if __name__ == '__main__':


    training_folder = sys.argv[1]
    test_folders = sys.argv[2:]

    colorlist = utils.get_colorlist()
    linestyles = utils.get_linestyles ()
    training_param_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    f = file(training_param_fn, 'r')
    print 'Loading training parameters from', training_param_fn
    training_params = json.load(f)
    
    ax = None
    
    P = Plotter()
    for i_, test_folder in enumerate(test_folders):
        test_param_fn = os.path.abspath(test_folder) + '/Parameters/simulation_parameters.json'
        f = file(test_param_fn, 'r')
        print 'Loading test parameters from', test_param_fn
        test_params = json.load(f)

        c = colorlist[(i_ + 1) % len(colorlist)]
        ls = linestyles[(i_ + 1) % len(linestyles)]
        legend_txt = 'Test w_mpn_D1(D2)=%.1f (%.1f)' % (test_params['mpn_d1_weight_amplification'], test_params['mpn_d2_weight_amplification'])
        stim_range = test_params['test_stim_range']
        ax = P.plot_xdisplacement(test_params, color=c, ls=ls, ax=ax, legend_label=legend_txt, plot_vertical_lines=False, stim_range=stim_range)

    P.set_training_params(training_params)
    ax = P.plot_xdisplacement(training_params, color=colorlist[0], ls='--', ax=ax, legend_label='Training', plot_vertical_lines=False, stim_range=stim_range)

    ax.legend(P.plots, P.legend_labels, loc='upper right')
    if len(test_folders) == 1:
        output_fn = test_params['figures_folder'] + 'comparison_training_test_xdisplacement.png'
    else:
        output_fn = training_params['figures_folder'] + 'comparison_training_test_xdisplacement.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
#    pylab.show()
