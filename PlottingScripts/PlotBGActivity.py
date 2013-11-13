import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import sys
import os
import utils
import re
import numpy as np
import pylab
import matplotlib


class ActivityPlotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        if it_max == None:
            self.it_max = self.params['n_iterations']
        else:
            self.it_max = it_max
        self.n_bins_x = 30
        self.n_x_ticks = 10
        self.x_ticks = np.linspace(0, self.n_bins_x, self.n_x_ticks)


    def plot_raster_simple(self):
        # first find files in Spikes folder 
        fn = self.params['spiketimes_folder_mpn'] + self.params['bg_spikes_fn_merged']
        print 'Loading spikes from:', fn
        d = np.loadtxt(fn)
        spikes = d[:, 1]
        gids = d[:, 0]
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(spikes, gids, 'o', markersize=3, color='k')

    def get_nspikes_interval(self, d, t0, t1):
        """
        d -- np.array containg the spike times (col 0 - gids, col 1 - times)
        """

        nspikes = np.zeros(self.params['n_exc_mpn'])
        for gid in xrange(1, self.params['n_exc_mpn'] + 1):
            cell_spikes = d[(d[:, 0] == gid).nonzero()[0], 1]
            idx = ((cell_spikes >= t0) == (cell_spikes <= t1)).nonzero()[0]
            nspikes[gid - 1] = idx.size
        return nspikes


if __name__ == '__main__':

    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)

    else:
        import simulation_parameters
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    print 'Merging spikes ...'
    utils.merge_and_sort_files(params['spiketimes_folder_mpn'] + params['bg_spikes_fn'], params['spiketimes_folder_mpn'] + params['bg_spikes_fn_merged'])
    Plotter = ActivityPlotter(params)#, it_max=1)
    Plotter.plot_raster_simple()
    pylab.show()
