import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from PlotMPNActivity import ActivityPlotter


import sys
import os
import utils
import re
import numpy as np
import pylab
import simulation_parameters
from FigureCreator import plot_params

if __name__ == '__main__':

    plot_params = {'backend': 'png',
                  'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'text.fontsize': 20,
                  'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'legend.pad': 0.2,     # empty space around the legend box
                  'legend.fontsize': 14,
                   'lines.markersize': 1,
                   'lines.markeredgewidth': 0.,
                   'lines.linewidth': 1,
                  'font.size': 12,
                  'path.simplify': False,
                  'figure.subplot.left':.15,
                  'figure.subplot.bottom':.15,
                  'figure.subplot.right':.94,
                  'figure.subplot.top':.84,
                  'figure.subplot.hspace':.30,
                  'figure.subplot.wspace':.30}

    pylab.rcParams.update(plot_params)

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])

#    stim_range = (0, 10)
#    n_stim = stim_range[1] - stim_range[0]
#    it_max = n_stim * params['n_iterations_per_stim']

    iter_range = (0, 3)
    it_max = 1
    AP = ActivityPlotter(params, it_max)
    AP.bin_spiketimes()
    compute_state_differences = False
    AP.plot_input(v_or_x='x')
    AP.plot_input(v_or_x='v')
    AP.plot_input_cmap(iteration=0)
    AP.plot_output(iter_range, v_or_x='x', compute_state_differences=compute_state_differences)
    AP.plot_output(iter_range, v_or_x='v', compute_state_differences=compute_state_differences)
    AP.plot_output_xv_cmap()
#    AP.plot_output(stim_range, v_or_x='v', compute_state_differences=compute_state_differences)
#    AP.plot_output(stim_range, v_or_x='gid', compute_state_differences=compute_state_differences)
    pylab.show()
