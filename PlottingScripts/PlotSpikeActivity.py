import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import simulation_parameters
import utils
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MergeSpikefiles
import FigureCreator
import plot_bcpnn_traces
import json
from MetaAnalysisClass import MetaAnalysisClass

class PlotSpikeActivity(MetaAnalysisClass):

    def __init__(self, argv, verbose=False):
        self.verbose = verbose
        self.rp_markersize = 2
        self.tick_interval = 8



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
                      'figure.subplot.left':.08,
                      'figure.subplot.bottom':.08,
                      'figure.subplot.right':.94,
                      'figure.subplot.top':.92,
                      'figure.subplot.hspace':.05, 
                      'figure.subplot.wspace':.30}
        #              'figure.figsize': get_fig_size(800)}
        plt.rcParams.update(plot_params)

        MetaAnalysisClass.__init__(self, argv, verbose) # call the constructor of the super/mother class
        # the constructer of the MetaAnalysisClass calls run_super_plot
        # with params and stim_range retrieved from the command line arguments


    def load_spikes(self):
        utils.merge_and_sort_files(self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn'], self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn_merged'])

        MS = MergeSpikefiles.MergeSpikefiles(self.params)
        if self.params['with_d2']:
            cell_types = ['d1', 'd2', 'action']#, 'supervisor']
        else:
            cell_types = ['d1', 'action']#, 'supervisor']

        for cell_type in cell_types:
            for i_action in range(self.params['n_actions']):
                merge_pattern = self.params['spiketimes_folder'] + self.params['%s_spikes_fn' % cell_type] + str(i_action) + '-' # '-' because NEST attaches something like -8357-0.dat to the file name
                output_fn = self.params['spiketimes_folder'] + self.params['%s_spikes_fn_merged' % cell_type] + str(i_action) + '.dat'
                if not os.path.exists(output_fn):
                    MS.merge_spiketimes_files(merge_pattern, output_fn)


    def run_super_plot(self, params, stim_range):
        self.params = params
        if stim_range == None:
            actions = np.loadtxt(self.params['actions_taken_fn'])
            n_stim = actions.shape[0]
            stim_range = range(0, n_stim)
        self.t_range = [0, 0]
        self.t_range[0] = stim_range[0] * self.params['t_iteration'] * self.params['n_iterations_per_stim']
        self.t_range[1] = (stim_range[-1] + 1) * self.params['t_iteration'] * self.params['n_iterations_per_stim']
        self.it_range = [0, 0]
        self.it_range[0] = stim_range[0] * self.params['n_iterations_per_stim']
        self.it_range[1] = (stim_range[-1] + 1) * self.params['n_iterations_per_stim']

        f = file(self.params['bg_gids_fn'], 'r')
        self.bg_gids = json.load(f)

        figsize = FigureCreator.get_fig_size(1400, portrait=False)
        self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(4, 1, height_ratios=(2, 1, 1, 1))
        self.load_spikes()
#        self.plot_mpn_rates()
        self.plot_bg_rates('action', n=0)
        self.plot_bg_rates('d2', n=1)
        self.plot_bg_rates('d1', n=2)



    def plot_bg_rates(self, cell_type, n=0):

        binsize = 25.
        n_cells_per_pop = self.params['n_cells_per_%s' % cell_type]
        n_bins = np.int((self.t_range[1] - self.t_range[0]) / binsize)
        ax0 = plt.subplot(self.gs[n])

        # build a color scheme
        # define the colormap
        cmap = matplotlib.cm.jet
        # extract all colors from the cmap
        cmaplist = [cmap(i) for i in xrange(cmap.N)]
        # force the first color entry to be grey #cmaplist[0] = (.5,.5,.5,1.0)
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        bounds = range(self.params['n_actions'])
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(np.arange(bounds[0], bounds[-1], 1.))
        rgba_colors = m.to_rgba(bounds)
        cb = self.fig.colorbar(m)
        cb.set_label('Action indices')#, fontsize=24)

        for i_action in xrange(self.params['n_actions']):
            print 'Plotting %s action %d' % (cell_type, i_action)
            print 'debug', self.params['spiketimes_folder'] + self.params['%s_spikes_fn_merged' % cell_type] + str(i_action) + '.dat'
            data = np.loadtxt(self.params['spiketimes_folder'] + self.params['%s_spikes_fn_merged' % cell_type] + str(i_action) + '.dat')
            if not os.path.getsize(self.params['spiketimes_folder'] + self.params['%s_spikes_fn_merged' % cell_type] + str(i_action) + '.dat') == 0:
                print 'no data in', cell_type, i_action
                if data.size == 2:
                    data = data.reshape((1, 2))
                hist, edges = np.histogram(data[:, 1], bins=n_bins, range=self.t_range)
                print 'hist', hist
                ax0.plot(edges[:-1] + .5 * binsize,  hist * (1000. / binsize) / n_cells_per_pop, lw=2, c=rgba_colors[i_action], label='%d' % i_action)
        ax0.set_ylabel('%s rate [Hz]' % (cell_type.capitalize()))
#        plt.legend()
        print 'finished'

    def plot_mpn_rates(self):

        raise NotImplementedError


if __name__ == '__main__':

#    MAC = MetaAnalysisClass(sys.argv)
    P = PlotSpikeActivity(sys.argv, verbose=True)
    plt.show()
