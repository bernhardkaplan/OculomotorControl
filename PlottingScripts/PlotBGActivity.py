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
import MergeSpikefiles


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

        self.rp_markersize = 3

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


    def plot_action_voltages(self):

        volt_fns = utils.find_files(self.params['spiketimes_folder_mpn'], 'bg_action_volt_')

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        print volt_fns
        for fn in volt_fns:
            path = self.params['spiketimes_folder_mpn'] + fn
            print 'path', path
            try:
                d = np.loadtxt(path)
                gids = np.unique(d[:, 0])
                for gid in gids:
                    time_axis, volt = utils.extract_trace(d, gid)
                    ax.plot(time_axis, volt, label='%d' % gid, lw=2)
            except:
                pass



    def plot_spikes_for_celltype(self, celltype, color='k', gid_offset=0, marker='o'):
        xa = -(self.params['t_sim']/10)
        recorder_type = 'spikes'
        mean = 0
        for naction in range(self.params['n_actions']):
            data = np.loadtxt(self.params['spiketimes_folder'] + str(naction) + celltype + '_merged_' + recorder_type + '.dat')
            if len(data)<2:
                print 'no data in', celltype, naction
            else:
                data[:, 0] += gid_offset
                pylab.plot(data[:,1], data[:,0], linestyle='None', marker='o', c=color, markeredgewidth=0, markersize=self.rp_markersize)
#                pylab.plot(data[:,1], data[:,0], linestyle='None', marker=marker, c=color, markeredgewidth=0, markersize=self.rp_markersize)
                mean += (np.min(data[:,0]) + np.max(data[:,0]))/2
        mean = mean / self.params['n_actions']
        pylab.text(xa, mean, celltype, color=color)
        print 'mean', mean


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
    if params['training']:
        cell_types = ['d1', 'd2', 'actions', 'supervisor']
    else:
        cell_types = ['d1', 'd2', 'actions']
    cell_types_volt = ['d1', 'd2', 'actions']

    print 'nstates ', params['n_states'], 'nactions ', params['n_actions']
    MS = MergeSpikefiles.MergeSpikefiles(params)
    for cell_type in cell_types:
        print 'Merging spiketimes file for %s ' % (cell_type)
        for naction in range(params['n_actions']):
            print 'Merging spiketimes file for %d ' % (naction)
            merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type] + str(naction)
            output_fn = params['spiketimes_folder'] + str(naction) + params['%s_spikes_fn_merged' % cell_type] 
            MS.merge_spiketimes_files(merge_pattern, output_fn)

    Plotter = ActivityPlotter(params)#, it_max=1)
    colors = ['b','g', 'r', 'c', 'm', 'y', 'k']
    markers = ['|', '-', 'o', 'D', '+', '4', 'v', 's']

    offset = 0
    for z, cell_type in enumerate(cell_types):
        cl = colors[z % len(colors)]
        marker = markers[z % len(colors)]
        print 'debug', cl, marker, cell_type
#        if cell_type == 'd2':
#            offset = -params['n_cells_D1']
        Plotter.plot_spikes_for_celltype(cell_type, color=cl, gid_offset=offset, marker=marker)

#    Plotter.plot_raster_simple()
#    Plotter.plot_action_voltages()
#    Plotter.plot_action_spikes()
    pylab.show()
