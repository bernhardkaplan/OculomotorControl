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

        self.n_bins_y = 200
        self.n_y_ticks = 10
        self.y_ticks = np.linspace(0, self.n_bins_y, self.n_y_ticks)
        self.load_tuning_prop()
        self.n_cells = self.params['n_exc_mpn']
        self.spiketrains = [[] for i in xrange(self.n_cells)]
        self.d = {}


    def load_tuning_prop(self):
        print 'ActivityPlotter.load_tuning_prop ...'
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_exc_fn'])
        self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])
        vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
        self.y_grid_x = np.linspace(0, 1, self.n_bins_y, endpoint=False)
        self.y_grid_vx = np.linspace(vmin, vmax, self.n_bins_y, endpoint=False)
        self.gid_to_posgrid_mapping_x = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 0], self.y_grid_x)
        self.gid_to_posgrid_mapping_vx = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 2], self.y_grid_vx)



    def plot_input(self, v_or_x='x'):
        if v_or_x == 'x':
            y_grid = self.y_grid_x
        else:
            y_grid = self.y_grid_vx
        d = np.zeros((y_grid.size, self.it_max)) 

        for iteration in xrange(self.it_max):
            print 'Plot input iteration', iteration
            fn_to_match = (self.params['input_nspikes_fn_mpn'] + 'it%d_' % iteration).rsplit('/')[-1]
            list_of_files = utils.find_files(self.params['input_folder_mpn'], fn_to_match)
            for fn_ in list_of_files:
                fn = self.params['input_folder_mpn'] + fn_
                print 'Loading:', fn
                d_it = np.loadtxt(fn, dtype=np.int)
                nspikes = d_it[:, 1]

                for i_gid in xrange(d_it[:, 0].size):
                    gid = d_it[i_gid, 0]
                    # map the gids to their position in the xgrid
#                    ypos = self.gid_to_posgrid_mapping_vx[gid, 1]
                    ypos = self.gid_to_posgrid_mapping_x[gid, 1]
                    d[xpos, iteration] += d_it[i_gid, 1]
#                    if d_it[i_gid, 1] > 0:
#                        print 'gid %d xpos %d nspikes_input %d' % (gid, xpos, d_it[i_gid, 1])
                    

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
        cbar.set_label('Output rate [Hz]')

        ylabels = ['%.1f' % (float(xtick) / self.n_bins_y) for xtick in self.x_ticks]
        ax.set_yticks(self.y_ticks)
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
        print 'debug n_bins_time', n_bins_time
        self.nspikes_binned = np.zeros((self.n_cells, n_bins_time)) # binned activity over time
        for gid in xrange(self.n_cells):
            nspikes = len(self.spiketrains[gid])
            if (nspikes > 0):
                count, bins = np.histogram(self.spiketrains[gid], bins=n_bins_time, range=(0, self.params['t_sim']))
                self.nspikes_binned[gid, :] = count
        return self.nspikes_binned

    def plot_output(self, stim_range=(0, 1), v_or_x='x', compute_state_differences=None):
        
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

        n_iter = self.params['n_iterations_per_stim'] * stim_range[1] - stim_range[0]
        iter_range = (self.params['n_iterations_per_stim'] * stim_range[0], self.params['n_iterations_per_stim'] * stim_range[1])

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
            ylabels = ['%.1f' % (float(xtick) / self.n_bins_y) for xtick in self.y_ticks]
            ax.set_yticks(self.y_ticks)
            ax.set_yticklabels(ylabels)

        if v_or_x == 'v':
            vmin, vmax = np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])
            ylabels = ['%.1f' % (float(xtick) * (vmax - vmin) / self.n_bins_y + vmin) for xtick in self.y_ticks]
            ax.set_yticks(self.y_ticks)
            ax.set_yticklabels(ylabels)

        output_fn = self.params['data_folder'] + 'mpn_output_activity_%s-sorting.dat' % (v_or_x)
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, self.d[v_or_x])

        output_fig = self.params['figures_folder'] + 'mpn_output_activity_%s-sorting.png' % (v_or_x)
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig)


        if compute_state_differences:
            fig = pylab.figure()
            ax_vec_dist = fig.add_subplot(111)
            ax_vec_dist.set_xlabel('Iteration diff index')
            ax_vec_dist.set_ylabel('State vector distance differece')
            ax_vec_dist.plot(range(n_iter - 1), vector_distance_difference_seq, marker='o', markersize=4)



    def plot_retinal_displacement(self, stim_range=None, ax=None, lw=3, c='b'):
        if stim_range == None:
            if self.params['training']:
                if self.params['n_stim'] == 1:
                    stim_range = [0, 1]
                else:
                    stim_range = range(self.params['n_stim'])
            else:
                stim_range = self.params['test_stim_range']
        print 'plot_retinal_displacement loads:', self.params['motion_params_fn']
        d = np.loadtxt(self.params['motion_params_fn'])
        it_min = stim_range[0] * self.params['n_iterations_per_stim']
        it_max = (stim_range[-1] + 1) * self.params['n_iterations_per_stim']
        t_axis = d[it_min:it_max, 4]
        t_axis += .5 * self.params['t_iteration']
#        print 'debug', t_axis.shape, d.shape, it_min, it_max
#        d[:, 4] = t_axis
#        x_displacement = np.abs(d[it_min:it_max, 0] - .5)
        x_displacement = d[it_min:it_max, 0] - .5
#        x_displacement = np.zeros(it_max - it_min)
        output_fn = self.params['data_folder'] + 'mpn_xdisplacement.dat'
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, d)

        if ax == None:
            fig = pylab.figure()
            ax = fig.add_subplot(111)

        for stim in xrange(stim_range[0], stim_range[-1] + 1):
            it_start_stim = stim * self.params['n_iterations_per_stim']
            it_stop_stim = (stim + 1) * self.params['n_iterations_per_stim'] - 1
            x_displacement_stim = d[it_start_stim:it_stop_stim, 0] - .5
#            x_displacement_stim = np.abs(d[it_start_stim:it_stop_stim, 0] - .5)
#            x_displacement[it_start_stim:it_stop_stim] = x_displacement_stim
            for it_ in xrange(self.params['n_iterations_per_stim'] - self.params['n_silent_iterations']):
                it_1 = it_ + stim * self.params['n_iterations_per_stim']
                it_2 = it_ + stim * self.params['n_iterations_per_stim'] + 2
                p1, = ax.plot(t_axis[it_1:it_2], x_displacement[it_1:it_2], lw=lw, c=c)
#                p1, = ax.plot(t_axis[it_1:it_2], x_displacement[it_1:it_2], lw=lw, c=c)
#            ax.plot((stim + 1) * self.param['n_iterations_per_stim'], 

#        ax.plot(t, x_displacement, lw=3)

        if self.params['training']:
            ax.set_title('Training')
        else:
            ax.set_title('Testing')
        self.plot_vertical_lines(ax, show_iterations=True)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Retinal displacement (x-dim)')
        t0 = it_min * self.params['t_iteration']
        t1 = it_max * self.params['t_iteration']
        ax.set_xlim((t0, t1))
#        ax.set_ylim((0, 1.2))
        output_fig = self.params['figures_folder'] + 'mpn_displacement.png'
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig)
        return (t_axis, x_displacement)


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
                        ax.annotate(str(int(action)), xy=(t0 + .2 * t_scale, ymin + (ymax - ymin) * .95), color='r')
                    it_cnt += 1


    def plot_raster_sorted(self, title='', cell_type='exc', sort_idx=0, t_range=None):
        """
        sort_idx : the index in tuning properties after which the cell gids are to be sorted for  the rasterplot
        """
        if cell_type == 'exc':
            tp = self.tuning_prop_exc
        else:
            tp = self.tuning_prop_ing

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
            ax.plot((xlim[0], xlim[1]), (.5, .5), ls='--', lw=3, c='k')
        elif sort_idx == 2:
            ax.plot((xlim[0], xlim[1]), (.0, .0), ls='--', lw=3, c='k')
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
        self.plot_vertical_lines(ax)


    def plot_training_sequence(self):
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

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Perceived stimuli')
        ax.set_ylabel('Iteration within one training stimulus')
        ax.set_xlabel('Start x-position')

        motion_params = np.loadtxt(self.params['motion_params_fn'])
        n_stim = motion_params[:, 0].size
        mp = np.zeros((n_stim, 5))
        for i in xrange(n_stim):
            mp[i, :] = motion_params[i, :]
            mp[i, 1] = i
#            print 'Debug stim and motion_params', i, mp[i, :]
            ax.annotate('(%.2f, %.2f)' % (mp[i, 0], mp[i, 2]), (max(0, mp[i, 0] - .1), mp[i, 1] + .2))
        
        ax.quiver(mp[:, 0], mp[:, 1], mp[:, 2], mp[:, 3], \
                  angles='xy', scale_units='xy', scale=1, headwidth=4, pivot='tail')#, width=0.007)

        xmin = mp[np.argmin(mp[:, 2]), 0] + mp[np.argmin(mp[:, 2]), 2] - .5
        xmax = mp[np.argmax(mp[:, 2]), 0] + mp[np.argmax(mp[:, 2]), 2] + .5
        ax.plot((.5 ,.5), (0, n_stim), ls='--', c='k')
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((-.5, n_stim + 0.5))

        return fig




class MetaAnalysisClass(object):

    def __init__(self, argv, plot_training_folder=None):
        print 'Argv:', len(argv)
        stim_range = None

        # optional: plot the training data
        if plot_training_folder != None:
            training_params = utils.load_params(plot_training_folder)

        if len(argv) == 1: # plot current parameters
            print '\nPlotting only stim 1!\n\n'
            network_params = simulation_parameters.global_parameters()  
            params = network_params.params
            utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])
            print '\nPlotting the default parameters give in simulation_parameters.py\n'
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
                stim_range = (int(argv[1]), int(argv[2]))
                network_params = simulation_parameters.global_parameters()  
                params = network_params.params
                utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])
                print '\nPlotting the default parameters give in simulation_parameters.py\n'
                self.run_single_folder_analysis(params, stim_range)
            else:
                self.run_analysis_for_folders(argv[1:], training_params=training_params)
        elif len(argv) == 4: #  PlotMPNActivity [FOLDER] [STIM_1] [STIM_2]
            folder_name = argv[1]
            if argv[2].isdigit() and argv[3].isdigit():
                stim_range = (int(argv[2]), int(argv[3]))
                params = utils.load_params(folder_name)
                utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])
                (x_data, y_data) = self.run_xdisplacement_analysis(params, stim_range)
                self.run_single_folder_analysis(params, stim_range)
                self.run_xdisplacement_analysis(params, stim_range)
            else:
                self.run_analysis_for_folders(argv[1:], training_params=training_params, stim_range=stim_range)
        elif len(argv) > 4: #  PlotMPNActivity [FOLDER_1] [FOLDER_2] .... [FOLDER_N]
            # do the same operation for many folders
            self.run_analysis_for_folders(argv[1:], training_params=training_params, stim_range=stim_range)


    def run_xdisplacement_analysis(self, params, stim_range):
        Plotter = ActivityPlotter(params)#, it_max=1)
        (t_axis, x_displacement) = Plotter.plot_retinal_displacement(stim_range=stim_range)
        return (t_axis, x_displacement)


    def run_single_folder_analysis(self, params, stim_range):
        Plotter = ActivityPlotter(params)#, it_max=1)
#        fig = Plotter.plot_training_sequence()
#        output_fn = params['figures_folder'] + 'training_sequence.png'
#        print 'Saving to', output_fn
#        fig.savefig(output_fn)

#        Plotter.plot_input()
        Plotter.bin_spiketimes()
        Plotter.plot_output()

        if stim_range != None:
            t_range = [0, 0]
            t_range[0] = stim_range[0] * params['t_iteration'] * params['n_iterations_per_stim']
            t_range[1] = stim_range[1] * params['t_iteration'] * params['n_iterations_per_stim']
        else:
            t_range = None

        # plot x - pos sorting
        print 'Plotting raster plots'
        fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by x-position', sort_idx=0, t_range=t_range)
        if params['debug_mpn']:
            Plotter.plot_input_spikes_sorted(ax, sort_idx=0)
        output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_xpos.png'
        print 'Saving to', output_fn
        fig.savefig(output_fn)

        # plot vx - sorting
        fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by preferred speed', sort_idx=2, t_range=t_range)
        if params['debug_mpn']:
            Plotter.plot_input_spikes_sorted(ax, sort_idx=2)
        output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_vx.png'
        print 'Saving to', output_fn
        fig.savefig(output_fn)

        del Plotter

    def run_analysis_for_folders(self, folders, training_params=None, stim_range=None):
        """
        folders -- list of folders with same time/data parameters (wrt to data size to be analysed)


        """
        if stim_range == None:
            stim_range = (0, 5) # to be changed

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
