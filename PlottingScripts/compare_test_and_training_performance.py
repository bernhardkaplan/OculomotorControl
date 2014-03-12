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

    def plot_xdisplacement(self, params, color='k', ls='-', ax=None, legend_label='', plot_vertical_lines=True, it_max=None):

        if ax == None:
            fig = pylab.figure()
            ax = fig.add_subplot(111)
        ax.set_title('Comparison of training vs. test performance')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Retinal displacement')
        fn = params['data_folder'] + 'mpn_xdisplacement.dat'
        print 'Loading data:', fn
        d = np.loadtxt(fn)

        xdispl = np.abs(d[:, 0] - .5)
        t_axis = d[:, 4]
        t_axis += .5 * params['t_iteration']
        n_iterations = d[:, 0].size
        if it_max == None:
            it_max = params['n_stim']
        for stim in xrange(it_max):
            for it_ in xrange(params['n_iterations_per_stim'] - 2):
                it_1 = it_ + stim*params['n_iterations_per_stim']
                it_2 = it_ + stim*params['n_iterations_per_stim'] + 2
                p1, = ax.plot(t_axis[it_1:it_2], xdispl[it_1:it_2], c=color, lw=3, ls=ls)

#        p1, = ax.plot(t_axis, xdispl, c=color, lw=3, ls=ls)
        self.plots.append(p1)
        self.legend_labels.append(legend_label)

        if plot_vertical_lines:
            AP = ActivityPlotter(params)
            AP.plot_vertical_lines(ax)

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
        legend_txt = 'Test w_mpn_bg=%.1f' % (test_params['mpn_bg_weight_amplification'])
        it_max = test_params['n_stim']
        ax = P.plot_xdisplacement(test_params, color=c, ls=ls, ax=ax, legend_label=legend_txt, plot_vertical_lines=False, it_max=it_max)

    P.set_training_params(training_params)
    ax = P.plot_xdisplacement(training_params, color=colorlist[0], legend_label='Training', plot_vertical_lines=True, it_max=it_max)

    ax.legend(P.plots, P.legend_labels, loc='upper right')
    output_fn = training_params['figures_folder'] + 'comparison_training_test_xdisplacement.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=300)
    pylab.show()
