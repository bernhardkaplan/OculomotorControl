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

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])

    stim_range = (0, 10)
    n_stim = stim_range[1] - stim_range[0]
    it_max = n_stim * params['n_iterations_per_stim']
    AP = ActivityPlotter(params, it_max)
    AP.bin_spiketimes()
    compute_state_differences = False
    AP.plot_output(stim_range, v_or_x='x', compute_state_differences=compute_state_differences)
    AP.plot_output(stim_range, v_or_x='v', compute_state_differences=compute_state_differences)
    AP.plot_output(stim_range, v_or_x='gid', compute_state_differences=compute_state_differences)
    pylab.show()
