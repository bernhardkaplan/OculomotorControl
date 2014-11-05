import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import simulation_parameters
import utils
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MergeSpikefiles
import FigureCreator
import plot_bcpnn_traces
import json
from MetaAnalysisClass import MetaAnalysisClass


class PlotEverything(MetaAnalysisClass):

    def __init__(self, argv, verbose=False):
        self.verbose = verbose
        self.rp_markersize = 2
        self.tick_interval = 8
        MetaAnalysisClass.__init__(self, argv, verbose) # call the constructor of the super/mother class
        # the constructer of the MetaAnalysisClass calls run_super_plot
        # with params and stim_range retrieved from the command line arguments


    def run_super_plot(self, params, stim_range):
        self.params = params

        utils.merge_spikes(params)
        print 'run_super_plot: folder %s, stim_range' % params['folder_name'], stim_range

        if stim_range == None:
            actions = np.loadtxt(self.params['actions_taken_fn'])
            n_stim = actions.shape[0]
            stim_range = range(0, n_stim)

        t_range = [0, 0]
        t_range[0] = stim_range[0] * self.params['t_iteration'] * self.params['n_iterations_per_stim']
        t_range[1] = (stim_range[-1] + 1) * self.params['t_iteration'] * self.params['n_iterations_per_stim']

        self.it_range = [0, 0]
        self.it_range[0] = stim_range[0] * self.params['n_iterations_per_stim']
        self.it_range[1] = (stim_range[-1] + 1) * self.params['n_iterations_per_stim']

        print 'xlim:', t_range

        figsize = FigureCreator.get_fig_size(1400, portrait=False)
        self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(4, 1, height_ratios=(2, 2, 1, 1))

        self.plot_bg_spikes(t_range)
        self.plot_mpn_spikes(t_range)
        self.plot_retinal_displacement_and_reward(stim_range, t_range)
        output_fn = self.params['figures_folder'] + 'super_plot_%d_%d.png' % (stim_range[0], stim_range[-1])
        print 'Saving figure to:', output_fn
        plt.savefig(output_fn)


    def plot_bg_spikes(self, t_range):

        ax0 = plt.subplot(self.gs[0])

        marker = 'o'
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        cell_types = ['d1', 'd2', 'actions']
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

        if t_range != None:
            ax0.set_xlim(t_range)

        # set ylim
        gid_min, gid_max = np.infty, -np.infty
        f = file(self.params['bg_gids_fn'], 'r')
        gids = json.load(f)
        for ct in cell_types:
#            print np.min(gids[ct]), np.max(gids[ct])
#            min(
            gid_min = min(gid_min, np.min(gids[ct]))
            gid_max = max(gid_max, np.max(gids[ct]))

#            gid_min, gid_max = min(min(gids[ct], gid_min), max(max[ct]), gid_max)
        print 'gid_min, gid_max:', gid_min, gid_max
        ax0.set_ylim((gid_min, gid_max))
        self.plot_vertical_lines(ax0)
        self.set_xticks(ax0, tick_interval=self.tick_interval)

        ax0.set_ylabel('BG cells')


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
        
        ax1 = plt.subplot(self.gs[1])
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
            ax1.set_ylabel('Receptive field position')
        if sort_idx == 1:
            ax1.set_ylim((np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])))
            ax1.set_ylabel('Preferred speed')

        self.plot_vertical_lines(ax1)
        self.set_xticks(ax1, tick_interval=self.tick_interval)


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




    def plot_retinal_displacement_and_reward(self, stim_range, t_range):
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

        rewards = actions[:, 3] #np.loadtxt(self.params['rewards_given_fn'])
#        print 'rewards', rewards.shape
#        if len(rewards) == 1: # only one stimulus
#            rewards = rewards.reshape((1, 1))
        K_vec = np.loadtxt(self.params['K_values_fn']) 
        ax2 = plt.subplot(self.gs[2])
        ax3 = plt.subplot(self.gs[3])
        color = 'b'
        for i_stim in xrange(stim_range[0], stim_range[-1]):
            t0 = i_stim * self.params['t_iteration'] * self.params['n_iterations_per_stim'] + 1 * self.params['t_iteration'] # + 1 because stimulus appears in iteration 1 (not 0) within a stimulus
            t1 = i_stim * self.params['t_iteration'] * self.params['n_iterations_per_stim'] + 2 * self.params['t_iteration'] # + 2 for the consequence of the action
            t0 += .5 * self.params['t_iteration'] # shift to the middle of the iteration
            t1 += .5 * self.params['t_iteration']
            x_stim = mp[i_stim, 0]
            v_eye = actions[i_stim, 0]
            x_after = utils.get_next_stim(self.params, mp[i_stim, :], actions[i_stim, 0])[0]
            r_test = utils.get_reward_from_perceived_states(x_stim, x_after)
            ax2.plot(np.array([t0, t1]), np.array([x_stim, x_after]), color=color, lw=3)

        ax2.set_ylim((-.05, 1.05))
        self.plot_vertical_lines(ax2)
        self.set_xticks(ax2, tick_interval=self.tick_interval)


        # plot the K_vec
        idx0 = stim_range[0] * self.params['n_iterations_per_stim']
        idx1 = (stim_range[-1] + 1) * self.params['n_iterations_per_stim']
        ax3.plot(range(idx0, idx1), K_vec[idx0:idx1])

        ax2.set_xlim(t_range)
        ax2.set_ylabel('Retinal\ndisplacement')

        ax3.set_xlim((self.it_range[0], self.it_range[1]))
        ax3.set_ylabel('Reward')

        ylim = ax3.get_ylim()
        # plot vertical lines
        for it_ in xrange(self.it_range[0], self.it_range[1]):
            ax3.plot((it_, it_), (ylim[0], ylim[1]), '--', lw=1, c='k')

        # plot horizontal line for retinal displacement
        xlim2 = ax2.get_xlim()
        ax2.plot((xlim2[0], xlim2[1]), (.5, .5), ls='-', c='k')


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

if __name__ == '__main__':

#    MAC = MetaAnalysisClass(sys.argv)
    P = PlotEverything(sys.argv, verbose=True)
    plt.show()
