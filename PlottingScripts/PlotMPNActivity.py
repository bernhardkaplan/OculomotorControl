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
import matplotlib
#matplotlib.use('Agg')
import pylab
import simulation_parameters
import matplotlib
from FigureCreator import plot_params
pylab.rcParams.update(plot_params)
from matplotlib import cm

class ActivityPlotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        if it_max == None:
            self.it_max = self.params['n_iterations']
        else:
            self.it_max = it_max

        self.n_bins_v = 50 # number of bins for activity colormap 
        self.n_bins_x = 50
        self.n_y_ticks = 10
        self.n_x_ticks = 10
        self.load_tuning_prop()
        self.n_cells = self.params['n_exc_mpn']
        self.spiketrains = [[] for i in xrange(self.n_cells)]
        self.d = {}
        self.spiketimes_binned = False
#        self.training_stimuli = np.loadtxt(self.params['training_stimuli_fn'])

    def load_tuning_prop(self):
        print 'ActivityPlotter.load_tuning_prop ...'
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_exc_fn'])
        vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
        self.y_grid_x = np.linspace(0, 1, self.n_bins_x, endpoint=False)
        self.y_grid_vx = np.linspace(vmin, vmax, self.n_bins_v, endpoint=False)
        self.gid_to_posgrid_mapping_x = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 0], self.y_grid_x)
        self.gid_to_posgrid_mapping_vx = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 2], self.y_grid_vx)



    def plot_input_cmap(self, iteration=0, stim_params=None, t_plot=None):
        if stim_params == None:
            stim_params = self.params['initial_state']
        x_grid = self.y_grid_x
        y_grid = self.y_grid_vx
        gid_to_posgrid_mapping_x = self.gid_to_posgrid_mapping_x
        gid_to_posgrid_mapping_y = self.gid_to_posgrid_mapping_vx
        xlabel = '$x_{preferred}$'
        ylabel = '$v_x$'

        d = np.zeros((self.n_bins_x, self.n_bins_v))
        nspikes_thresh = 0
        ncells_above_nspikes_thresh = np.zeros((self.n_bins_x, self.n_bins_v))
        for fn in os.listdir(self.params['input_folder_mpn']):
            m = re.match('input_spikes_%d_(\d+).dat' % iteration, fn)
            if m:
                gid = int(m.groups()[0])
                fn_ = self.params['input_folder_mpn'] + fn
#                print 'DEBUG Loading:', fn
                dspikes = np.loadtxt(fn_)
                nspikes = dspikes.size
                ypos = gid_to_posgrid_mapping_x[gid, 1]
                xpos = gid_to_posgrid_mapping_y[gid, 1]
                d[xpos, ypos] += nspikes
                if nspikes > nspikes_thresh:
                    ncells_above_nspikes_thresh[xpos, ypos] += 1

        for i_ in xrange(x_grid.size):
            for j_ in xrange(y_grid.size):
                if ncells_above_nspikes_thresh[i_, j_] > 0:
                    d[i_, j_] /= ncells_above_nspikes_thresh[i_, j_]

        if t_plot != None:
            d /= t_plot / 1000.

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(d)#, cmap='binary')

        title = 'Input spikes for $x_{stim}=%.2f\ v_{stim}=%.2f$\n $\\beta_X=%.2f\ \\beta_V=%.2f$' % \
                (stim_params[0], stim_params[2], self.params['blur_X'], self.params['blur_V'])

        ax.set_title(title)
        ax.set_xlim((0, d.shape[0]))
        ax.set_ylim((0, d.shape[1]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = pylab.colorbar(cax)
        cbar.set_label('Input rate [Hz]')

        x_ticks = np.linspace(0, self.n_bins_x, self.n_x_ticks)
        xlabels = ['%.1f' % (float(xtick) / self.n_bins_x) for xtick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xlabels)

        y_ticks = np.linspace(0, self.n_bins_v, self.n_y_ticks)
        vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
        ylabels = ['%.1f' % (float(ytick) * (vmax - vmin) / self.n_bins_v + vmin) for ytick in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(ylabels)

        output_fig = 'mpn_input_spike_distribution_stim_x%.2f_v%.2f_blurX%.2f_bV%.2f.png' % \
                (stim_params[0], stim_params[2], self.params['blur_X'], self.params['blur_V'])
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig, dpi=200)


                

    def plot_input(self, iteration=0, v_or_x='x'):
        if v_or_x == 'x':
            y_grid = self.y_grid_x
            n_bins_y = self.n_bins_x
        else:
            y_grid = self.y_grid_vx
            n_bins_y = self.n_bins_v

        d = np.zeros((y_grid.size, self.it_max)) 

#        for iteration in xrange(self.it_max):

        for fn in os.listdir(self.params['input_folder_mpn']):
            m = re.match('input_spikes_%d_(\d+).dat' % iteration, fn)
            if m:
                gid = int(m.groups()[0])
                fn_ = self.params['input_folder_mpn'] + fn
#                print 'DEBUG Loading:', fn
                dspikes = np.loadtxt(fn_)
                nspikes = dspikes.size
                ypos = self.gid_to_posgrid_mapping_x[gid, 1]
                d[ypos, iteration] += nspikes

        d /= self.params['t_iteration'] / 1000.
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(d)#, cmap='binary')

        if self.params['training']:
            testtraining = 'training'
        else:
            testtraining = 'testing'


        ax.set_title('Input spikes during %s clustered by x-pos' % testtraining)
        ax.set_ylim((0, d.shape[0]))
        ax.set_xlim((0, d.shape[1]))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('x-pos')
        cbar = pylab.colorbar(cax)
        cbar.set_label('Input rate [Hz]')

#        ylabels = ['%.1f' % (float(xtick) / n_bins_y) for xtick in y_grid]
#        y_ticks = np.linspace(0, n_bins_y, self.n_y_ticks)
#        ax.set_yticks(y_ticks)
#        ax.set_yticklabels(ylabels)

        if v_or_x == 'x':
            n_bins_y = self.n_bins_x
            y_ticks = np.linspace(0, n_bins_y, self.n_y_ticks)
            ylabels = ['%.1f' % (float(xtick) / n_bins_y) for xtick in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(ylabels)

        if v_or_x == 'v':
            n_bins_y = self.n_bins_v
            y_ticks = np.linspace(0, n_bins_y, self.n_y_ticks)
            vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
            ylabels = ['%.1f' % (float(xtick) * (vmax - vmin) / n_bins_y + vmin) for xtick in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(ylabels)


        output_fn = self.params['data_folder'] + 'mpn_input_activity.dat'
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, d)

        output_fig = self.params['figures_folder'] + 'mpn_input_activity.png'
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig)


    def get_nspikes_interval(self, d, t0, t1):
        """
        d -- np.array containg the spike times (col 0 - gids, col 1 - times)
        """

        nspikes = np.zeros(self.params['n_exc_mpn'])
        for gid in xrange(1, self.params['n_exc_mpn'] + 1):
            cell_spikes = d[(d[:, 0] == gid).nonzero()[0], 1]
            idx = ((cell_spikes >= t0) == (cell_spikes <= t1)).nonzero()[0]
            nspikes[gid - 1] = idx.size
#            print 'cell %d spikes between %d - %d: %d times' % (gid, t0, t1, nspikes[gid - 1])
        return nspikes


    def bin_spiketimes(self):
        merged_spike_fn = self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn_merged']
        if not os.path.exists(merged_spike_fn):
            utils.merge_and_sort_files(self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn'], merged_spike_fn)
        self.nspikes, self.spiketrains = utils.get_spikes(merged_spike_fn, n_cells=self.n_cells, get_spiketrains=True, gid_idx=0)

        n_bins_time = self.params['n_iterations']
        self.nspikes_binned = np.zeros((self.n_cells, n_bins_time)) # binned activity over time
        for gid in xrange(self.n_cells):
            nspikes = len(self.spiketrains[gid])
            if (nspikes > 0):
                count, bins = np.histogram(self.spiketrains[gid], bins=n_bins_time, range=(0, self.params['t_sim']))
                self.nspikes_binned[gid, :] = count
        self.spiketimes_binned = True
        return self.nspikes_binned


    def plot_output_xv_cmap(self, stim_params=None, t_plot=None):
        """
        Requires bin_spiketimes() to be called before
        """
        assert (self.spiketimes_binned == True), 'Please call bin_spiketimes() before calling plot_output_xv_map'
        if stim_params == None:
            stim_params = self.params['initial_state']
        x_grid = self.y_grid_x
        gid_to_posgrid_mapping_x = self.gid_to_posgrid_mapping_x
        xlabel = '$x_{preferred}$'
        y_grid = self.y_grid_vx

        gid_to_posgrid_mapping_y = self.gid_to_posgrid_mapping_vx
        ylabel = '$v_x$'

        nspikes_thresh = 0
        d = np.zeros((self.n_bins_x, self.n_bins_v))
        ncells_above_nspikes_thresh = np.zeros((self.n_bins_x, self.n_bins_v))
        for gid in xrange(self.params['n_exc_mpn']):
#            print 'DEBUG gid %d tp_x = %.2f gid_to_posgrid_mapping_x:' % (gid, self.tuning_prop_exc[gid, 0]), gid_to_posgrid_mapping_x[gid, 1]
            ypos = gid_to_posgrid_mapping_x[gid, 1]
            xpos = gid_to_posgrid_mapping_y[gid, 1]
            nspikes = len(self.spiketrains[gid])
            d[xpos, ypos] += nspikes
            if nspikes > nspikes_thresh:
                ncells_above_nspikes_thresh[xpos, ypos] += 1

        for i_ in xrange(x_grid.size):
            for j_ in xrange(y_grid.size):
                if ncells_above_nspikes_thresh[i_, j_] > 0:
                    d[i_, j_] /= ncells_above_nspikes_thresh[i_, j_]

        if t_plot != None:
            d /= t_plot / 1000.

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(d)#, cmap='binary')

        title = 'Output spikes for $x_{stim}=%.2f\ v_{stim}=%.2f$\n $\\beta_X=%.2f\ \\beta_V=%.2f$' % \
                (stim_params[0], stim_params[2], self.params['blur_X'], self.params['blur_V'])
        ax.set_title(title)
        ax.set_xlim((0, d.shape[0]))
        ax.set_ylim((0, d.shape[1]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = pylab.colorbar(cax)
        cbar.set_label('Output rate [Hz]')

#        y_ticks = np.linspace(0, 10, self.n_y_ticks)
#        ylabels = ['%.1f' % (float(tick) / self.n_bins_v) for tick in y_ticks]
#        ax.set_yticks(y_ticks)
#        ax.set_yticklabels(ylabels)

        x_ticks = np.linspace(0, self.n_bins_x, self.n_x_ticks)
        xlabels = ['%.1f' % (float(xtick) / self.n_bins_x) for xtick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xlabels)

        y_ticks = np.linspace(0, self.n_bins_v, self.n_y_ticks)
        vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
        ylabels = ['%.1f' % (float(ytick) * (vmax - vmin) / self.n_bins_v + vmin) for ytick in y_ticks]
#        ylabels = ['%.1f' % (float(self.tuning_prop_exc[i_, 2])) for i_ in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(ylabels)

        output_fig = 'mpn_output_spike_distribution_stim_x%.2f_v%.2f_blurX%.2f_bV%.2f.png' % \
                (stim_params[0], stim_params[2], self.params['blur_X'], self.params['blur_V'])
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig, dpi=200)



    def plot_output(self, iter_range=(0, 1), v_or_x='x', compute_state_differences=None):
        assert (self.spiketimes_binned == True), 'Please call bin_spiketimes() before calling plot_output!'
        
        # adjust for VX  /  X - plotting and training, testing
        if self.params['training']:
            testtraining = 'training'
        else:
            testtraining = 'testing'
        if v_or_x == 'x':
            y_grid = self.y_grid_x
            gid_to_posgrid_mapping = self.gid_to_posgrid_mapping_x
            title = 'Output spikes during %s binned & sorted by x-pos' % testtraining
            ylabel = '$x_{preferred}$'
        elif v_or_x == 'v':
            y_grid = self.y_grid_vx
            gid_to_posgrid_mapping = self.gid_to_posgrid_mapping_vx
            title = 'Output spikes during %s binned & sorted by $v_x$' % testtraining
            ylabel = '$v_x$'
        else: # use the gid as index
            y_grid = np.arange(self.n_cells, dtype=np.int)
            gid_to_posgrid_mapping = np.zeros((self.n_cells, 2))
            gid_to_posgrid_mapping[:, 0] = np.arange(self.n_cells, dtype=np.int)
            gid_to_posgrid_mapping[:, 1] = gid_to_posgrid_mapping[:, 0]
            title = 'Output spikes during %s' % testtraining
            ylabel = 'GID'

        n_iter = iter_range[1] - iter_range[0]

        self.d[v_or_x] = np.zeros((y_grid.size, n_iter))
        nspikes_thresh = 0
        for iteration in xrange(iter_range[0], iter_range[1]):
            print 'Plot output iteration', iteration
            cells_per_grid_cell = np.zeros(y_grid.size) # how many cells have been above a threshold activity during this iteration
            for gid in xrange(self.params['n_exc_mpn']):
                ypos = gid_to_posgrid_mapping[gid, 1]
                self.d[v_or_x][ypos, iteration] += self.nspikes_binned[gid, iteration]

                if self.nspikes[gid] > nspikes_thresh:
                    cells_per_grid_cell[ypos] += 1
            for grid_idx in xrange(y_grid.size):
                if cells_per_grid_cell[grid_idx] > 0:
                    self.d[v_or_x][grid_idx, iteration] /= cells_per_grid_cell[grid_idx]
        self.d[v_or_x] /= self.params['t_iteration'] / 1000.

        if compute_state_differences:
            # compute the state vector distance differences for two consecutive / sequent iterations
            # and the matrix for distance differences between all iterations
            vector_distance_difference_seq = np.zeros(n_iter - 1)       
            vector_distance_difference_matrix = np.zeros((n_iter, n_iter))
            for i_ in xrange(iter_range[0], iter_range[1]):
                for j_ in xrange(iter_range[0], iter_range[1]):
                    vector_distance_difference_matrix[i_, j_] = utils.distance(self.d[v_or_x][:, i_], self.d[v_or_x][:, j_])
                if i_ != iter_range[1] - 1:
                    vector_distance_difference_seq[i_] = vector_distance_difference_matrix[i_, i_ + 1]
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            cax = ax.pcolormesh(vector_distance_difference_matrix)#, cmap='binary')
            ax.set_title('State vector difference matrix for %s vectors' % v_or_x)
            ax.set_ylim((0, vector_distance_difference_matrix.shape[0]))
            ax.set_xlim((0, vector_distance_difference_matrix.shape[1]))
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Iteration')
            cbar = pylab.colorbar(cax)
            cbar.set_label('State vector difference')


        fig = pylab.figure()
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(self.d[v_or_x])#, cmap='binary')

        ax.set_title(title)
        ax.set_ylim((0, self.d[v_or_x].shape[0]))
        ax.set_xlim((0, self.d[v_or_x].shape[1]))
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        cbar = pylab.colorbar(cax)
        cbar.set_label('Output rate [Hz]')

        if v_or_x == 'x':
            n_bins_y = self.n_bins_x
            y_ticks = np.linspace(0, n_bins_y, self.n_y_ticks)
            ylabels = ['%.1f' % (float(xtick) / n_bins_y) for xtick in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(ylabels)

        if v_or_x == 'v':
            n_bins_y = self.n_bins_v
            y_ticks = np.linspace(0, n_bins_y, self.n_y_ticks)
            vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
            ylabels = ['%.1f' % (float(xtick) * (vmax - vmin) / n_bins_y + vmin) for xtick in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(ylabels)

        output_fn = self.params['data_folder'] + 'mpn_output_activity_%s-sorting.dat' % (v_or_x)
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, self.d[v_or_x])

        output_fig = self.params['figures_folder'] + 'mpn_output_activity_%s-sorting.png' % (v_or_x)
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig, dpi=200)


        if compute_state_differences:
            fig = pylab.figure()
            ax_vec_dist = fig.add_subplot(111)
            ax_vec_dist.set_xlabel('Iteration diff index')
            ax_vec_dist.set_ylabel('State vector distance differece')
            ax_vec_dist.plot(range(n_iter - 1), vector_distance_difference_seq, marker='o', markersize=4)



    def plot_retinal_displacement(self, stim_range=None, ax=None, lw=3, c='b'):
        return (None, None)
        pass
        
    """
        print 'DEBUG plot_retinal_displacement stim_range:', stim_range
        if stim_range == None:
            if self.params['n_stim'] == 1:
                stim_range = [0, 1]
            else:
                stim_range = range(self.params['n_stim'])
            stim_range_label = [stim_range[0], stim_range[1]]
        elif not self.params['training']:
            stim_range_label = [stim_range[0], stim_range[1]]
            stim_range[0] -= self.params['test_stim_range'][0]
            stim_range[1] -= self.params['test_stim_range'][0]
        else:
            stim_range_label = [stim_range[0], stim_range[1]]
        n_stim = stim_range[-1] - stim_range[0]
        if self.params['training']:
            fn = self.params['motion_params_training_fn']
        else:
            fn = self.params['motion_params_testing_fn']
        print 'plot_retinal_displacement loads:', fn
        d = np.loadtxt(fn)
        it_min = stim_range[0] * self.params['n_iterations_per_stim']
        it_max = (stim_range[-1] + 1) * self.params['n_iterations_per_stim']
        print 'debug stim_range:', stim_range, 'it min max', it_min, it_max
        t_axis = d[it_min:it_max, 4]
        t_axis += .5 * self.params['t_iteration']
        x_displacement = d[it_min:it_max, 0] - .5
        output_fn = self.params['data_folder'] + 'mpn_xdisplacement.dat'
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, d)

        if ax == None:
            fig = pylab.figure()
            ax = fig.add_subplot(111)

        ylim = (-.5, .5)
        for stim, stim_ in enumerate(range(stim_range[0], stim_range[-1] + 1)):
            it_start_stim = stim * self.params['n_iterations_per_stim']
            it_stop_stim = (stim + 1) * self.params['n_iterations_per_stim'] - self.params['n_silent_iterations']
            x_displacement_stim = d[it_start_stim:it_stop_stim, 0] - .5
            print 'x_displacement_stim:', x_displacement_stim, self.params['n_iterations_per_stim'] - self.params['n_silent_iterations']# - 1
            print 't_axis:', t_axis
#            if n_stim == 1:
#                for it_ in xrange(self.params['n_iterations_per_stim'] - self.params['n_silent_iterations']):
#                    it_1 = it_ 
#                    it_2 = it_ + 2
#                    p1, = ax.plot(t_axis[it_1:it_2], x_displacement[it_1:it_2], 'o', ms=5, c=c)
#            else:

            for it_ in xrange(self.params['n_iterations_per_stim'] - self.params['n_silent_iterations'] - 1):
                it_1 = it_ + stim * self.params['n_iterations_per_stim']
                it_2 = it_ + stim * self.params['n_iterations_per_stim'] + 2
                print 'debug it_ ', it_1, it_2, '\tt_axis:', t_axis[it_1:it_2], x_displacement_stim[it_1:it_2]
                print 'debug x_displ stim', x_displacement_stim
                print 'debug x_displ ', x_displacement
                p1, = ax.plot(t_axis[it_1:it_2], x_displacement[it_1:it_2], lw=lw, c=c)
#                p1, = ax.plot(t_axis[it_1:it_2], x_displacement_stim[it_1:it_2], lw=lw, c=c)

        if self.params['training']:
            if self.params['reward_based_learning']:
                ax.set_title('Reward based learning')
            else:
                ax.set_title('Training')
        else:
            ax.set_title('Testing')
        ax.set_ylim(ylim)
        self.plot_vertical_lines(ax, show_iterations=True)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Retinal displacement (x-dim)')
        t0 = it_min * self.params['t_iteration']
        t1 = it_max * self.params['t_iteration']
        ax.plot((t0, t1), (0., 0.), c='k', lw=2, ls=':')

        if self.params['training'] and self.params['reward_based_learning']:
            try:
                self.plot_reward(ax)
            except:
                print '\nPlotMPNActivity failed to plot rewards!\n File %s exists?\n\t%s\n' % (self.params['rewards_given_fn'], os.path.exists(self.params['rewards_given_fn']))

        ax.set_xlim((t0, t1))
        output_fig = self.params['figures_folder'] + 'mpn_displacement_%d-%d.png' % (stim_range_label[0], stim_range_label[1])
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig, dpi=200)
        return (t_axis, x_displacement)
    """


    def plot_reward(self, ax):
        rewards = np.loadtxt(self.params['rewards_given_fn'])
        ax2 = ax.twinx()
        ms_min = 2
        ms_max = 20.
        reward_max = np.max(rewards)
        for i_ in xrange(rewards.size):
            if rewards[i_] > 0:
                c = 'r'
                s = '^'
                fillstyle = 'full'
            else:
                c = 'b'
                s = 'v'
                fillstyle = 'full' #'none'
            ms = np.abs(np.round(rewards[i_] / reward_max * ms_max)) + ms_min
            print 'rewards i_ markersize', rewards[i_], i_, ms
            ax2.plot(i_ * self.params['t_iteration'] + .5 * self.params['t_iteration'], rewards[i_], s, markersize=ms, c=c, fillstyle=fillstyle)


    def plot_vertical_lines(self, ax, params=None, time_scale=True, show_iterations=True):
        """
        if time_scale == True: the x-axis units is ms
        if time_scale == False: the x-axis units is iterations
        """
        if params == None:
            params = self.params
        (ymin, ymax) = ax.get_ylim()
        it_cnt = 0
        if time_scale:
            t_scale = params['t_iteration']
        else:
            t_scale = 1
        
        try:
            actions_taken = np.loadtxt(params['actions_taken_fn'])
            for i_stim in xrange(params['n_stim']):
                t0 = i_stim * params['n_iterations_per_stim'] * t_scale
                ax.plot((t0, t0), (ymin, ymax), ls='-', lw=2, c='k')
                if show_iterations:
                    for it_ in xrange(params['n_iterations_per_stim']):
                        t0 = it_ * t_scale + i_stim * params['n_iterations_per_stim'] * t_scale
                        ax.plot((t0, t0), (ymin, ymax), ls='-.', c='k')
                        ax.annotate(str(it_cnt), xy=(t0 + .45 * t_scale, ymin + (ymax - ymin) * .05))

                        it_idx = it_ + i_stim * params['n_iterations_per_stim'] + 1 # + 1 because the first row stores the 'initial action'
                        action = actions_taken[it_idx, 2]
                        if not np.isnan(action) != 0.:
                            ax.annotate('%.1f' % (action), xy=(t0 + .2 * t_scale, ymin + (ymax - ymin) * .95), color='r', fontsize=8)
                        it_cnt += 1
        except:
            print 'PlotMPNActivity: Failed to plot_vertical_lines!\nFile %s exists:\n\t%s' % (params['actions_taken_fn'], os.path.exists(params['actions_taken_fn']))



    def plot_raster_sorted(self, title='', cell_type='exc', sort_idx=0, t_range=None):
        """
        sort_idx : the index in tuning properties after which the cell gids are to be sorted for  the rasterplot
        """
        if cell_type == 'exc':
            tp = self.tuning_prop_exc
        else:
            tp = self.tuning_prop_inh

        tp_idx_sorted = tp[:, sort_idx].argsort() # + 1 because nest indexing

        merged_spike_fn = self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn_merged']
        print 'Plotter.plot_raster_sorted loads:', merged_spike_fn
        spikes_unsrtd = np.loadtxt(merged_spike_fn)
        if t_range != None:
            spikes_unsrtd = utils.get_spiketimes_within_interval(spikes_unsrtd, t_range[0], t_range[1])

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel('Time [ms]')
        if sort_idx == 0:
            ax.set_ylabel('Cells sorted by RF-position')
        elif sort_idx == 2:
            ax.set_ylabel('Cells sorted by preferred speed')
        for i_, gid in enumerate(tp_idx_sorted):
            spikes = utils.get_spiketimes(spikes_unsrtd, gid + 1)
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * tp[gid, sort_idx]
            ax.plot(spikes, y_, 'o', markersize=3, color='k')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if sort_idx == 0:
            ax.set_ylim((-.02, 1.02))
        elif sort_idx == 2:
            ax.set_ylim((tp[:, 2].min() - 0.02, tp[:, 2].max() + 0.02))

        if sort_idx == 0:
            ax.plot((xlim[0], xlim[1]), (.5, .5), ls='--', lw=3, c='r')
        elif sort_idx == 2:
            ax.plot((xlim[0], xlim[1]), (.0, .0), ls='--', lw=3, c='r')
        if t_range != None:
            ax.set_xlim(t_range)

        return fig, ax


    def plot_input_spikes_sorted(self, ax=None, title='', sort_idx=0):
        """
        Input spikes are stored in seperate files for each cell and for each iteration.
        --> filenames are Folder/InputSpikes_MPN/input_spikes_X_GID.dat X = iteration, GID = cell gid
        """
        print 'plotting input spikes ...'

        if ax == None:
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            ax.set_title(title)
            print ' DEBUG None '
        tp = self.tuning_prop_exc
        tp_idx_sorted = tp[:, sort_idx].argsort() # + 1 because nest indexing

        for fn in os.listdir(self.params['input_folder_mpn']):
            m = re.match('input_spikes_(\d+)_(\d+).dat', fn)
            if m:
                iteration = int(m.groups()[0])
                gid = int(m.groups()[1])
                y_pos_of_cell = tp[gid, sort_idx]
                fn_ = self.params['input_folder_mpn'] + fn
                d = np.loadtxt(fn_)
                ax.plot(d, y_pos_of_cell * np.ones(d.size), 'o', markersize=3, alpha=.1, color='b')
        del d
        self.plot_vertical_lines(ax)


    def plot_training_sequence(self, stim_range=None):
        """
        plots the 1D training sequence

         ^ stimulus
         | number
         |
         |      ->
         |    <----
         |  ->
         +---------------->
            x-start-pos
        """
        if stim_range == None:
            if self.params['training']:
                if self.params['n_stim'] == 1:
                    stim_range = [0, 1]
                else:
                    stim_range = range(self.params['n_stim'])
            else:
                stim_range = self.params['test_stim_range']

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Perceived stimuli')
        ax.set_ylabel('Iteration within one training stimulus')
        ax.set_xlabel('Start x-position')

        motion_params = np.loadtxt(self.params['motion_params_fn'])
        n_stim = stim_range[1] - stim_range[0]
        
        mp = np.zeros((n_stim, 5))
        for i in xrange(stim_range[0], stim_range[1]):
            print 'debug', i, motion_params.shape, mp[i, :]
            print 'debug', i, motion_params.shape, motion_params[i, :]
            mp[i, :] = motion_params[i, :]
            mp[i, 1] = i
            ax.annotate('(%.2f, %.2f)' % (mp[i, 0], mp[i, 2]), (max(0, mp[i, 0] - .1), mp[i, 1] + .2))
        
        ax.quiver(mp[:, 0], mp[:, 1], mp[:, 2], mp[:, 3], \
                  angles='xy', scale_units='xy', scale=1, headwidth=4, pivot='tail')#, width=0.007)

        xmin = mp[np.argmin(mp[:, 2]), 0] + mp[np.argmin(mp[:, 2]), 2] - .5
        xmax = mp[np.argmax(mp[:, 2]), 0] + mp[np.argmax(mp[:, 2]), 2] + .5
        ax.plot((.5 ,.5), (0, n_stim), ls='--', c='k')
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((-.5, n_stim + 0.5))

        output_fn = self.params['figures_folder'] + 'training_sequence_%d-%d.png' % (stim_range[0], stim_range[1])
        print 'Saving to', output_fn
        fig.savefig(output_fn, dpi=200)




class MetaAnalysisClass(object):
    """
    Depening on the arguments passed to the constructor,
    different plot functions are called
    """

    def __init__(self, argv, plot_training_folder=None, show=False):
        print 'Argv:', len(argv), argv
        stim_range = None

        # optional: plot the training data
        if plot_training_folder != None:
            training_params = utils.load_params(plot_training_folder)

        if len(argv) == 1: # plot current parameters
            print '\nPlotting the default parameters given in simulation_parameters.py\n'
            print '\nPlotting only stim 1!\n\n'
            network_params = simulation_parameters.global_parameters()  
            params = network_params.params
            utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])
            self.run_single_folder_analysis(params, stim_range)
            (x_data, y_data) = self.run_xdisplacement_analysis(params, stim_range)
        elif len(argv) == 2: # PlotMPNActivity [FOLDER]
            folder_name = argv[1]
            params = utils.load_params(folder_name)
            utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])
            self.run_single_folder_analysis(params, stim_range)
            (x_data, y_data) = self.run_xdisplacement_analysis(params, stim_range)
        elif len(argv) == 3: #  PlotMPNActivity [STIM_1] [STIM_2]
            if argv[1].isdigit() and argv[2].isdigit():
                stim_range = [int(argv[1]), int(argv[2])]
                network_params = simulation_parameters.global_parameters()  
                params = network_params.params
                utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])
                print '\nPlotting the default parameters give in simulation_parameters.py\n'
                self.run_single_folder_analysis(params, stim_range)
                (x_data, y_data) = self.run_xdisplacement_analysis(params, stim_range)
                for i_ in xrange(stim_range[0], stim_range[1]):
                    (x_data, y_data) = self.run_xdisplacement_analysis(params, [i_, i_+1])
                    self.run_single_folder_analysis(params, [i_, i_ + 1])
            else:
                self.run_analysis_for_folders(argv[1:], training_params=training_params)
        elif len(argv) == 4: #  PlotMPNActivity [FOLDER] [STIM_1] [STIM_2]
            folder_name = argv[1]
            if argv[2].isdigit() and argv[3].isdigit():
                stim_range = [int(argv[2]), int(argv[3])]
                params = utils.load_params(folder_name)
                utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])
                for i_ in xrange(stim_range[0], stim_range[1]):
                    (x_data, y_data) = self.run_xdisplacement_analysis(params, [i_, i_+1])
                    self.run_single_folder_analysis(params, [i_, i_ + 1])
                (x_data, y_data) = self.run_xdisplacement_analysis(params, stim_range)
#                self.run_single_folder_analysis(params, stim_range)
            else:
                self.run_analysis_for_folders(argv[1:], training_params=training_params, stim_range=stim_range)
        elif len(argv) > 4: #  PlotMPNActivity [FOLDER_1] [FOLDER_2] .... [FOLDER_N]
            # do the same operation for many folders
            self.run_analysis_for_folders(argv[1:], training_params=training_params, stim_range=stim_range)

        if show:
            pylab.show()

    def run_xdisplacement_analysis(self, params, stim_range):
        Plotter = ActivityPlotter(params)#, it_max=1)
        (t_axis, x_displacement) = Plotter.plot_retinal_displacement(stim_range=stim_range)
        return (t_axis, x_displacement)


    def run_single_folder_analysis(self, params, stim_range):
        Plotter = ActivityPlotter(params)#, it_max=1)
#        Plotter.plot_training_sequence(stim_range)
#        Plotter.plot_input()
#        Plotter.bin_spiketimes()
#        Plotter.plot_output()

        if not params['training'] and (stim_range != None):
            stim_range_label = [stim_range[0], stim_range[1]]
            stim_range[0] -= params['test_stim_range'][0]
            stim_range[1] -= params['test_stim_range'][0]
        elif stim_range != None:
            stim_range_label = [stim_range[0], stim_range[1]]

        if stim_range != None:
            t_range = [0, 0]
            t_range[0] = stim_range[0] * params['t_iteration'] * params['n_iterations_per_stim']
            t_range[1] = stim_range[1] * params['t_iteration'] * params['n_iterations_per_stim']
            output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_xpos_%03d-%03d.png' % (stim_range_label[0], stim_range_label[1])
        else:
            t_range = None
            output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_xpos.png' 

        # plot x - pos sorting
        print 'Plotting raster plots'
        title = 'Exc cells sorted by x-position'
#        title = 
        fig, ax = Plotter.plot_raster_sorted(title=title, sort_idx=0, t_range=t_range)
        if params['debug_mpn']:
            Plotter.plot_input_spikes_sorted(ax, sort_idx=0)
        print 'Saving to', output_fn
        fig.savefig(output_fn, dpi=200)

        # plot vx - sorting
        fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by preferred speed', sort_idx=2, t_range=t_range)
        if params['debug_mpn']:
            Plotter.plot_input_spikes_sorted(ax, sort_idx=2)
        if stim_range != None:
            output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_vx_%d-%d.png' % (stim_range_label[0], stim_range_label[1])
        else:
            output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_vx.png' 
        print 'Saving to', output_fn
        fig.savefig(output_fn, dpi=200)

        del Plotter


    def run_analysis_for_folders(self, folders, training_params=None, stim_range=None):
        """
        folders -- list of folders with same time/data parameters (wrt to data size to be analysed)


        """
        if stim_range == None:
            stim_range = [0, 5] # to be changed

        n_stim = stim_range[1] - stim_range[0]
        # load the first parameter set to determine data size
        params = utils.load_params(folders[0])
        n_idx_per_stim = params['n_iterations_per_stim'] * n_stim
        n_folders = len(folders)
        all_data = np.zeros((n_folders, n_idx_per_stim))

        for i_, folder in enumerate(folders):
            params = utils.load_params(folder)
            (x_data, y_data) = self.run_xdisplacement_analysis(params, stim_range)
            self.run_single_folder_analysis(params, stim_range)
            all_data[i_, :] = y_data

        no_plot_idx = [(i + 1) * params['n_iterations_per_stim'] - 2 for i in xrange(params['n_stim'])]
        no_plot_idx += [(i + 1) * params['n_iterations_per_stim'] - 1 for i in xrange(params['n_stim'])]
        ax = self.plot_collected_data(all_data, x_data, no_plot_idx=no_plot_idx)

        if training_params != None:
            Plotter = ActivityPlotter(training_params)#, it_max=1)
            (t_axis, x_displacement) = Plotter.plot_retinal_displacement(stim_range=stim_range, ax=ax, lw=3, c='k')




    def plot_collected_data(self, d, x_data, no_plot_idx=False):
        """
        d :
                    2-dimensional array for collected data, 
                    first dimension  -> index of curve
                    2nd dim          -> index within the curve
        x_data :
                    1-dimensional array (e.g. time axis) which is common to all curves
        no_plot_idx :
                    list of indices which shall not be plotted
        """

        n_curves = d[:, 0].size
        n_idx_per_curve = d[0, :].size
        average_curve = np.zeros((n_idx_per_curve, 2))

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for idx in xrange(n_idx_per_curve):
            average_curve[idx, 0] = d[:, idx].mean()
            average_curve[idx, 1] = d[:, idx].std()


        norm = matplotlib.mpl.colors.Normalize(vmin=0, vmax=n_curves)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)

        if no_plot_idx == False:
            ax.errorbar(x_data, average_curve[:, 0], yerr=average_curve[:, 1], lw=3, c='r')
            for i_ in xrange(n_curves):
                ax.plot(x_data, d[i_, :])
        else:
            for i_ in xrange(n_curves):
                for idx in xrange(n_idx_per_curve-2):
                    if idx not in no_plot_idx:
                        c = m.to_rgba(i_)
                        ax.plot(x_data[idx:idx+2], d[i_, idx:idx+2], ls='-', c=c, alpha=0.5)
#            for idx in xrange(n_idx_per_curve-1):
#                if idx not in no_plot_idx:
#                    ax.errorbar(x_data[idx:idx+1], average_curve[idx:idx+1, 0], yerr=average_curve[idx:idx+1], lw=3, c='r')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Retinal displacement')

        wtf = 0.5
        ax.set_title('Test performance averaged over %d trials, WTF=%.1f' % (n_curves, wtf))

        output_fn = 'average_xdisplacement_WTF_%.1f.png'  % (wtf)
        print 'Saving to:', output_fn
        pylab.savefig(output_fn, dpi=200)
        return ax



if __name__ == '__main__':

    """
    Usage:
        python scriptname.py [FOLDER_NAME or .json parameter file] [ITERATION_0] [ITERATION_1]
        [ ] -- optional arguments
        

    """

#    if len(sys.argv) > 1:
#        params = utils.load_params(sys.argv[1])
#    else:
#        param_tool = simulation_parameters.global_parameters()
#        params = param_tool.params


#    training_folder = 'Training_Cluster_taup90000_nStim400_it15-90000_wMPN-BG3.00_bias1.00_K1.00/'
    training_folder = None
    MAC = MetaAnalysisClass(sys.argv, plot_training_folder=training_folder)
#    pylab.show()
