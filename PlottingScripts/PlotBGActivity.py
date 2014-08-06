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
matplotlib.use('Agg')
import pylab
import MergeSpikefiles
import PlotMPNActivity


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
        fn = self.params['spiketimes_folder'] + self.params['bg_spikes_fn_merged']
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


    def plot_voltages(self, cell_type, action_idx=None, output_fn=None):

        if action_idx != None:
            add = str(action_idx)
        else:
            add = ''
        if cell_type == 'actions':
            volt_fns = utils.find_files(self.params['spiketimes_folder'], 'bg_action_volt_' + add)
        elif cell_type == 'd1':
            volt_fns = utils.find_files(self.params['spiketimes_folder'], 'd1_volt_' + add)
        elif cell_type == 'd2':
            volt_fns = utils.find_files(self.params['spiketimes_folder'], 'd2_volt_' + add)


        fig = pylab.figure()
        ax = fig.add_subplot(111)
        print 'PlotBGActivity.plot_voltages found:', volt_fns
        for fn in volt_fns:
            path = self.params['spiketimes_folder'] + fn
            print 'plot_voltages loads', path
            try:
                d = np.loadtxt(path)
                gids = np.unique(d[:, 0])
                for gid in gids:
                    time_axis, volt = utils.extract_trace(d, gid)
                    ax.plot(time_axis, volt, label='%d' % gid, lw=2)
            except:
                pass

        if output_fn != None:
            pylab.savefig(output_fn)


    def plot_spikes_for_cell_type(self, cell_type, color='k', gid_offset=0, marker='o', ylim=None, ax=None, xlim=None):
        recorder_type = 'spikes'

        if ax == None:
            print 'creating figure'
            fig = pylab.figure()
            pylab.subplots_adjust(left=0.15)
            ax = fig.add_subplot(111)
        else:
            print 'no fig created'

        for naction in xrange(self.params['n_actions']):
            print 'Plotting %s action %d' % (cell_type, naction)
            data = np.loadtxt(self.params['spiketimes_folder'] + self.params['%s_spikes_fn_merged' % cell_type] + str(naction) + '.dat')
            if xlim != None and data.size > 2:
                data = utils.get_spiketimes_within_interval(data, xlim[0], xlim[1])
            if len(data)<2:
                print 'no data in', cell_type, naction
            elif data.size == 2:
                data[0] += gid_offset
                ax.plot(data[1], data[0], linestyle='None', marker=marker, c=color, markeredgewidth=0, markersize=self.rp_markersize)
            else:
                data[:, 0] += gid_offset
                ax.plot(data[:,1], data[:,0], linestyle='None', marker=marker, c=color, markeredgewidth=0, markersize=self.rp_markersize)

        if ylim != None:
            ax.set_ylim(ylim)
        if xlim != None:
            ax.set_xlim(xlim)
        mn, mx = utils.get_min_max_gids_for_bg(self.params, cell_type)
        ypos_label = .5 * (mx - mn) + mn
        xa = - 20 #(self.params['t_sim'] / 6.)
#        print 'DEBUG', ypos_label, cell_type, mn, mx, xa
        ax.text(xa, ypos_label, cell_type, color=color, fontsize=16)
        ax.set_xlabel('Time [ms]')
        if self.params['training']:
            ax.set_title('Spikes in BG during training')
        else:
            ax.set_title('Spikes in BG during testing, w_mpn_D1(D2)=%.2f (%.2f)' % (self.params['mpn_d1_weight_amplification'], self.params['mpn_d2_weight_amplification']))
        return ax


def run_plot_bg(params, stim_range):

    print 'Plotted the stim_range', stim_range
    print 'Merging spikes ...'
    if params['with_d2']:
        if params['training']:
            cell_types = ['d1', 'd2', 'actions']#, 'supervisor']
        else:
            cell_types = ['d1', 'd2', 'actions']#, 'supervisor']
    else:
        if params['training']:
            cell_types = ['d1', 'actions']#, 'supervisor']
        else:
            cell_types = ['d1', 'actions']#, 'supervisor']

    if stim_range == None:
        if params['training']:
            if params['n_stim'] == 1:
                stim_range = [0, 1]
            else:
                stim_range = range(params['n_stim'])
        else:
            stim_range = params['test_stim_range']
        xlim = None
    else:
        t_stim = params['n_iterations_per_stim'] * params['t_iteration']
        xlim = (stim_range[0] * t_stim, stim_range[1] * t_stim)

    MS = MergeSpikefiles.MergeSpikefiles(params)
    for cell_type in cell_types:
        for naction in range(params['n_actions']):
            merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type] + str(naction) + '-' # '-' because NEST attaches something like -8357-0.dat to the file name
            output_fn = params['spiketimes_folder'] + params['%s_spikes_fn_merged' % cell_type] + str(naction) + '.dat'
            if not os.path.exists(output_fn):
                MS.merge_spiketimes_files(merge_pattern, output_fn)

    Plotter = ActivityPlotter(params)#, it_max=1)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o']
#    markers = ['|', '-', 'o', 'D', '+', '4', 'v', 's']

    ax = None
    offset = 0
    gid_min, gid_max = np.infty, -np.infty
    for z, cell_type in enumerate(cell_types):
        mn, mx= utils.get_min_max_gids_for_bg(params, cell_type)
        gid_min, gid_max = min(mn, gid_min), max(mx, gid_max)
        cl = colors[z % len(colors)]
        marker = markers[z % len(markers)]
        ax = Plotter.plot_spikes_for_cell_type(cell_type, color=cl, gid_offset=offset, marker=marker, ylim=(gid_min, gid_max), ax=ax, xlim=xlim)

    PMPN = PlotMPNActivity.ActivityPlotter(params)
    PMPN.plot_vertical_lines(ax, params)
    if len(stim_range) > 1:
        output_fig = params['bg_rasterplot_fig'][:params['bg_rasterplot_fig'].rfind('.')] + 'wD1-D2_%.2f_%.2f_stim%d-%d.png' % \
                (params['mpn_d1_weight_amplification'], params['mpn_d2_weight_amplification'], stim_range[0], stim_range[1])
    else:
        output_fig = params['bg_rasterplot_fig'][:params['bg_rasterplot_fig'].rfind('.')] + 'wD1-D2_%.2f_%.2f_stim%d.png' % \
                (params['mpn_d1_weight_amplification'], params['mpn_d2_weight_amplification'], stim_range[0])
    print 'Saving figure to:', output_fig
    pylab.savefig(output_fig, dpi=200)

#    Plotter.plot_voltages('d1', action_idx=18, output_fn=params['figures_folder'] + 'd1_voltages.png')
#    Plotter.plot_raster_simple()
#    Plotter.plot_action_voltages()
#    Plotter.plot_action_spikes()

if __name__ == '__main__':

    stim_range = None
    if len(sys.argv) == 1:
        network_params = simulation_parameters.global_parameters()  
        params = network_params.params
        print '1\nPlotting the default parameters give in simulation_parameters.py\n'
        run_plot_bg(params, stim_range)
    elif len(sys.argv) == 2: # plot_ [FOLDER]
        folder_name = sys.argv[1]
        params = utils.load_params(folder_name)
        run_plot_bg(params, stim_range)
    elif len(sys.argv) == 3: #  plot_ [STIM_1] [STIM_2]
        if sys.argv[1].isdigit() and sys.argv[2].isdigit():
            stim_range = (int(sys.argv[1]), int(sys.argv[2]))
            network_params = simulation_parameters.global_parameters()  
            params = network_params.params
            print '\nPlotting the default parameters give in simulation_parameters.py\n'
        else:
            for fn in sys.argv[1:]:
                params = utils.load_params(fn)
                run_plot_bg(params, stim_range)
    elif len(sys.argv) == 4: #  PlotMPNActivity [FOLDER] [STIM_1] [STIM_2]
        folder_name = sys.argv[1]
        if sys.argv[2].isdigit() and sys.argv[3].isdigit():
            stim_range = (int(sys.argv[2]), int(sys.argv[3]))
            params = utils.load_params(folder_name)
            run_plot_bg(params, stim_range)
            for i_stim in xrange(stim_range[0], stim_range[1]):
                run_plot_bg(params, (i_stim, i_stim + 1))

        else:
            for fn in sys.argv[1:]:
                params = utils.load_params(fn)
                run_plot_bg(params, stim_range)
    elif len(sys.argv) > 4: #  PlotMPNActivity [FOLDER_1] [FOLDER_2] .... [FOLDER_N]
        for fn in sys.argv[1:]:
            params = utils.load_params(fn)
            run_plot_bg(params, stim_range)

#    pylab.show()
