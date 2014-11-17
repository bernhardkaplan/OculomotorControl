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
import json
from MetaAnalysisClass import MetaAnalysisClass


class PlotEverything(MetaAnalysisClass):

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


    def run_super_plot(self, params, stim_range):
        self.fig_cnt = 0
        self.params = params
        # load bg gids
        f = file(self.params['bg_gids_fn'], 'r')
        self.bg_gids = json.load(f)

        utils.merge_spikes(params)
        print 'run_super_plot: folder %s, stim_range' % params['folder_name'], stim_range

        if stim_range == None:
            if self.params['training']:
                actions = np.loadtxt(self.params['actions_taken_fn'])
                n_stim = actions.shape[0]
                stim_range = range(0, n_stim)
            else:
                stim_range = self.params['test_stim_range']

        t_range = [0, 0]
        t_range[0] = stim_range[0] * self.params['t_iteration'] * self.params['n_iterations_per_stim']
        t_range[1] = (stim_range[-1] + 1) * self.params['t_iteration'] * self.params['n_iterations_per_stim']

        self.it_range = [0, 0]
        self.it_range[0] = stim_range[0] * self.params['n_iterations_per_stim']
        self.it_range[1] = (stim_range[-1] + 1) * self.params['n_iterations_per_stim']

        print 'xlim:', t_range

        figsize = FigureCreator.get_fig_size(1400, portrait=False)
        self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(4, 1, height_ratios=(2, 1, 1, 1))
#        self.gs = gridspec.GridSpec(3, 1, height_ratios=(2, 1, 1))

        ax0 = self.plot_bg_spikes(t_range)
#        self.plot_mpn_spikes(t_range)
#        self.plot_bg_rates('action', t_range)

        if self.params['training']:
            trained_stim = utils.get_start_and_stop_iteration_for_stimulus_from_motion_params(self.params['motion_params_training_fn'])
            self.plot_trained_stim(ax0, trained_stim)

            self.plot_retinal_displacement(stim_range, t_range)
            
            self.plot_reward(stim_range, t_range)
        else:
            self.plot_retinal_displacement(stim_range, t_range)
        output_fn = self.params['figures_folder'] + 'super_plot_%d_%d.png' % (stim_range[0], stim_range[-1])
        print 'Saving figure to:', output_fn
        plt.savefig(output_fn, dpi=200)



    def plot_trained_stim(self, ax, trained_stim):
        """
        ax -- an axis element
        trained_stim -- is a dictionary with (x, v) as key and {'start': <int>, 'stop': <int>, 'cnt': <int> } as value, indicating 
        the start and stop iteration (line in the motion_params_fn) during which the stimulus has been trained.
        """
        ylim = ax.get_ylim()
#        if self.params['training']:
        stim_offset = utils.get_stim_offset(self.params)
        for (x, v) in trained_stim.keys():
            start, stop = trained_stim[(x, v)]['start'], trained_stim[(x, v)]['stop']
            cnt = trained_stim[(x, v)]['cnt']
            t_0 = start * self.params['n_iterations_per_stim'] * self.params['t_iteration']
            t_1 = stop * self.params['n_iterations_per_stim'] * self.params['t_iteration']

            text_pos_x = t_0 + 0.1 * (t_1 - t_0) 
            text_pos_y = ylim[1] + 0.04 * (ylim[1] - ylim[0])
            ax.text(text_pos_x, text_pos_y, '(%.2f, \n%.2f)\n%d: %d-%d' % (x, v, np.int(cnt + stim_offset), start, stop))
            ax.plot((t_0, t_0), (ylim[0], text_pos_y), ls='-', c='k', lw=3)
            ax.plot((t_1, t_1), (ylim[0], text_pos_y), ls='-', c='k', lw=3)


    def plot_bg_spikes(self, t_range):

        ax0 = plt.subplot(self.gs[self.fig_cnt])

        marker = 'o'
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        cell_types = ['d1', 'd2', 'action']
        for z_, cell_type in enumerate(cell_types):

            color = colors[z_ % len(colors)]
            for naction in xrange(self.params['n_actions']):
                fn = self.params['spiketimes_folder'] + self.params['%s_spikes_fn_merged' % cell_type] + str(naction) + '.dat'
                filesize = os.path.getsize(fn)
                if filesize > 0:
                    print 'Plotting %s action %d' % (cell_type, naction)
                    data = np.loadtxt(fn)
                    if t_range != None and data.size > 2:
                        data = utils.get_spiketimes_within_interval(data, t_range[0], t_range[1])
                    if len(data)<2:
                        print 'no data in', cell_type, naction
                    elif data.size == 2:
    #                    data[0] += gid_offset
                        ax0.plot(data[1], data[0], linestyle='None', marker=marker, c=color, markeredgewidth=0, markersize=self.rp_markersize)
                    else:
    #                    data[:, 0] += gid_offset
                        ax0.plot(data[:,1], data[:,0], linestyle='None', marker=marker, c=color, markeredgewidth=0, markersize=self.rp_markersize)

        bg_gid_ranges = utils.get_bg_gid_ranges(self.params)
        for z_, ct in enumerate(cell_types):
            gid_min, gid_max = bg_gid_ranges[ct]
            color = colors[z_ % len(colors)]
            text_pos_y = gid_min + .5 * (gid_max - gid_min)
            text_pos_x = t_range[0] - .02 * (t_range[1] - t_range[0])
#            print 'debug', ct, gid_min, gid_max, text_pos_x, text_pos_y
            ax0.text(text_pos_x, text_pos_y, '%s' % (ct.capitalize()), color=color, fontsize=16, rotation=90)

        if t_range != None:
            ax0.set_xlim(t_range)

        # set ylim
        gid_min, gid_max = np.infty, -np.infty
        ax0.set_ylim((np.min(bg_gid_ranges.values()), np.max(bg_gid_ranges.values())))

        self.plot_vertical_lines(ax0)
        self.set_xticks(ax0, tick_interval=self.tick_interval)

        ax0.set_ylabel('BG cells\n')#D1  D2  Actions')
        self.remove_xtick_labels(ax0)
        self.remove_ytick_labels(ax0)
        self.fig_cnt += 1
        return ax0


    def load_tuning_prop(self):
        print 'SuperPlot.load_tuning_prop ...'
        self.n_bins_y = 200
        self.n_y_ticks = 10
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_exc_fn'])
        vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
        self.y_grid_x = np.linspace(0, 1, self.n_bins_y, endpoint=False)
        self.y_grid_vx = np.linspace(vmin, vmax, self.n_bins_y, endpoint=False)
        self.gid_to_posgrid_mapping_x = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 0], self.y_grid_x)
        self.gid_to_posgrid_mapping_vx = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 2], self.y_grid_vx)


    def plot_mpn_spikes(self, t_range, sort_idx=0):

        self.load_tuning_prop()
        tp_idx_sorted = self.tuning_prop_exc[:, sort_idx].argsort() # + 1 because nest indexing
        
        ax1 = plt.subplot(self.gs[self.fig_cnt])
        merged_spike_fn = self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn_merged']
        print 'Plotter.plot_raster_sorted loads:', merged_spike_fn
        spikes_unsrtd = np.loadtxt(merged_spike_fn)
        if t_range != None:
            spikes_unsrtd = utils.get_spiketimes_within_interval(spikes_unsrtd, t_range[0], t_range[1])

        print 'Plotting MPN spikes sorted according to tuning properties'
        for i_, gid in enumerate(tp_idx_sorted):
            spikes = utils.get_spiketimes(spikes_unsrtd, gid + 1)
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * self.tuning_prop_exc[gid, sort_idx]
            ax1.plot(spikes, y_, 'o', markersize=self.rp_markersize, color='k')

        if self.params['debug_mpn']:
            self.plot_input_spikes_sorted(ax1, sort_idx=0)

        if t_range != None:
            ax1.set_xlim(t_range)

        if sort_idx == 0:
            ax1.set_ylim((0., 1.))
            ax1.set_ylabel('Receptive\nfield position')
        if sort_idx == 1:
            ax1.set_ylim((np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])))
            ax1.set_ylabel('Preferred speed')

        self.plot_vertical_lines(ax1)
        self.set_xticks(ax1, tick_interval=self.tick_interval)
        self.remove_xtick_labels(ax1)
        self.fig_cnt += 1
        return ax1


    def plot_input_spikes_sorted(self, ax, sort_idx=0):
        print 'Plotting input spikes...'

        tp = self.tuning_prop_exc
        tp_idx_sorted = tp[:, sort_idx].argsort()
        for fn in os.listdir(self.params['input_folder_mpn']):
            m = re.match('input_spikes_(\d+)_(\d+).dat', fn)
            if m:
                iteration = int(m.groups()[0])
                gid = int(m.groups()[1])
                y_pos_of_cell = tp[gid, sort_idx]
                fn_ = self.params['input_folder_mpn'] + fn
                d = np.loadtxt(fn_)
                ax.plot(d, y_pos_of_cell * np.ones(d.size), 'o', markersize=self.rp_markersize, alpha=.1, color='b')




    def plot_retinal_displacement(self, stim_range, t_range, plot_action_idx=True):
        print 'stim_range', stim_range
        if self.params['training']:
            fn = self.params['motion_params_training_fn']
        else:
            fn = self.params['motion_params_testing_fn']


        mp = np.loadtxt(fn)
        if len(mp.shape) == 1: # only one stimulus
            mp = mp.reshape((1, 4))

        actions = np.loadtxt(self.params['actions_taken_fn'])
        if len(actions.shape) == 1: # only one stimulus
            actions = actions.reshape((1, 4))

#        rewards = actions[:, 3] #np.loadtxt(self.params['rewards_given_fn'])
#        print 'rewards', rewards.shape
#        if len(rewards) == 1: # only one stimulus
#            rewards = rewards.reshape((1, 1))
        ax2 = plt.subplot(self.gs[self.fig_cnt])
        self.fig_cnt += 1

        color = 'b'
        ylim_ax2 = (-.05, 1.05)
        for i_stim in xrange(stim_range[0], stim_range[-1] + 1):
            t0 = i_stim * self.params['t_iteration'] * self.params['n_iterations_per_stim'] + 1 * self.params['t_iteration'] # + 1 because stimulus appears in iteration 1 (not 0) within a stimulus
            t1 = i_stim * self.params['t_iteration'] * self.params['n_iterations_per_stim'] + 2 * self.params['t_iteration'] # + 2 for the consequence of the action
            t0 += .5 * self.params['t_iteration'] # shift to the middle of the iteration
            t1 += .5 * self.params['t_iteration']
            x_stim = mp[i_stim, 0]
            v_eye = actions[i_stim, 0]
            action_idx = np.int(actions[i_stim, 2])
            x_after = utils.get_next_stim(self.params, mp[i_stim, :], v_eye)[0]
            r_test = utils.get_reward_from_perceived_states(x_stim, x_after)
            ax2.plot(np.array([t0, t1]), np.array([x_stim, x_after]), color=color, lw=3)
            if plot_action_idx:
                text_pos_x = t0 - 0.1 * self.params['t_iteration']
                ax2.text(text_pos_x, 0.95, '%d' % action_idx, color='k', fontsize=12)
                print 'Action %d:' % action_idx, self.bg_gids['action'][action_idx]

        ax2.set_ylim(ylim_ax2)
        self.plot_vertical_lines(ax2)
        self.set_xticks(ax2, tick_interval=self.tick_interval)
        self.remove_xtick_labels(ax2)
        ax2.set_xlim(t_range)
        ax2.set_ylabel('Retinal\ndisplacement')

        # plot horizontal line for retinal displacement
        xlim2 = ax2.get_xlim()
        ax2.plot((xlim2[0], xlim2[1]), (.5, .5), ls='-', c='k')
        return ax2

        # plot the K_vec with transitions
#        idx0 = stim_range[0] * self.params['n_iterations_per_stim']
#        idx1 = (stim_range[-1] + 1) * self.params['n_iterations_per_stim']
#        ax3.plot(range(idx0, idx1), K_vec[idx0:idx1])
#        ax3.set_xlim((self.it_range[0], self.it_range[1]))
#        ylim = ax3.get_ylim()
         #plot vertical lines
#        for it_ in xrange(self.it_range[0], self.it_range[1]):
#            ax3.plot((it_, it_), (ylim[0], ylim[1]), '--', lw=1, c='k')

    def plot_reward(self, stim_range, t_range, plot_action_idx=True):
        K_vec = np.loadtxt(self.params['K_values_fn']) 
        ax3 = plt.subplot(self.gs[self.fig_cnt])
        x_k = np.array([])
        y_k = np.array([])
        it_0, it_1 = stim_range[0] * self.params['n_iterations_per_stim'], (stim_range[-1] + 1) * self.params['n_iterations_per_stim']
        for it_ in xrange(it_0, it_1):
            x_ = np.arange(it_ * self.params['t_iteration'], (it_ + 1) * self.params['t_iteration'])
            y_ = K_vec[it_] * np.ones(self.params['t_iteration'])
            x_k = np.r_[x_k, x_]
            y_k = np.r_[y_k, y_]
        ax3.plot(x_k, y_k, c='k', lw=3)
        ax3.set_xlim((it_0 * self.params['t_iteration'], it_1 * self.params['t_iteration']))
        ylim_ax3 = (-np.max(np.abs(K_vec)) * 1.05, np.max(np.abs(K_vec)) * 1.05)
        ax3.set_ylim(ylim_ax3)

        ax3.set_ylabel('Reward')
        ax3.set_xlabel('Time [ms]')
        self.plot_vertical_lines(ax3)

        self.fig_cnt += 1
        return ax3

    def plot_bg_rates(self, cell_type, t_range):

        binsize = 10.
        n_cells_per_pop = self.params['n_cells_per_%s' % cell_type]
        n_bins = np.int((t_range[1] - t_range[0]) / binsize)
        ax0 = plt.subplot(self.gs[self.fig_cnt])

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
#            gids = self.bg_gids[cell_type][i_action]

            hist, edges = np.histogram(data[:, 1], bins=n_bins, range=t_range)
            ax0.plot(edges[:-1] + .5 * binsize,  hist * (1000. / binsize) / n_cells_per_pop, lw=2, c=rgba_colors[i_action], label='%d' % i_action)

        plt.legend()
        print 'finished'
        self.fig_cnt += 1


    def plot_vertical_lines(self, ax):

        (ymin, ymax) = ax.get_ylim()
        for it_ in xrange(self.it_range[0], self.it_range[1]):
            t0 = it_ * self.params['t_iteration']
            ax.plot((t0, t0), (ymin, ymax), ls='--', lw=1, c='k')


    def set_xticks(self, ax, tick_interval=1):
        old_xticks = ax.get_xticks()
        new_xticks = []

        for it_ in xrange(self.it_range[0], self.it_range[1], tick_interval):
            t0 = it_ * self.params['t_iteration']
            new_xticks.append(t0)
        ax.set_xticks(new_xticks)

    def remove_xtick_labels(self, ax):
        ax.set_xticklabels([])

    def remove_ytick_labels(self, ax):
        ax.set_yticklabels([])


if __name__ == '__main__':

#    MAC = MetaAnalysisClass(sys.argv)
    P = PlotEverything(sys.argv, verbose=True)
    plt.show()
