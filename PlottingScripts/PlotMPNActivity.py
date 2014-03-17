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
from FigureCreator import plot_params
pylab.rcParams.update(plot_params)
import simulation_parameters
import matplotlib
from matplotlib import cm

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
        self.load_tuning_prop()

    def load_tuning_prop(self):
        print 'ActivityPlotter.load_tuning_prop ...'
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_exc_fn'])
        self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])

        self.x_grid = np.linspace(0, 1, self.n_bins_x, endpoint=False)
        self.gid_to_posgrid_mapping = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 0], self.x_grid)



    def plot_input(self):
        d = np.zeros((self.it_max, self.x_grid.size)) #self.x_grid.size, self.it_max))

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
                    xpos = self.gid_to_posgrid_mapping[gid, 1]
                    d[iteration, xpos] += d_it[i_gid, 1]
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

        ax.set_title('Input spikes during %s' % testtraining)
        ax.set_ylim((0, d.shape[0]))
        ax.set_xlim((0, d.shape[1]))
        ax.set_ylabel('Iteration')
        ax.set_xlabel('x-pos')
        cbar = pylab.colorbar(cax)
        cbar.set_label('Input rate [Hz]')

        xlabels = ['%.1f' % (float(xtick) / self.n_bins_x) for xtick in self.x_ticks]
        ax.set_xticks(self.x_ticks)
        ax.set_xticklabels(xlabels)
        output_fn = self.params['data_folder'] + 'mpn_input_activity.dat'
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, d)

        output_fig = self.params['figures_folder'] + 'mpn_input_activity.png'
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig)

    
#    def get_nspikes(self, merged_spike_fn):
#        d = np.loadtxt(merged_spike_fn)
#        nspikes = np.zeros(self.params['n_exc_mpn'])
#        for gid in xrange(1, self.params['n_exc_mpn'] + 1):
#            idx = (d[:, 0] == gid).nonzero()[0]
#            nspikes[gid - 1] = idx.size
#        return nspikes


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


    def plot_output(self):
        merged_spike_fn = self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn_merged']
        utils.merge_and_sort_files(self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn'], merged_spike_fn)
        spike_data = np.loadtxt(merged_spike_fn)
        d = np.zeros((self.it_max, self.x_grid.size)) #self.x_grid.size, self.it_max))
        nspikes_thresh = 1
        for iteration in xrange(self.it_max):
            print 'Plot output iteration', iteration
            cells_per_grid_cell = np.zeros(self.x_grid.size) # how many cells have been above a threshold activity during this iteration
            t0, t1 = iteration * self.params['t_iteration'], (iteration + 1) * self.params['t_iteration']
            nspikes = self.get_nspikes_interval(spike_data, t0, t1) 
            for gid in xrange(self.params['n_exc_mpn']):
                xpos = self.gid_to_posgrid_mapping[gid, 1]
                d[iteration, xpos] += nspikes[gid]
                if nspikes[gid] > nspikes_thresh:
                    cells_per_grid_cell[xpos] += 1
            for grid_idx in xrange(self.x_grid.size):
                if cells_per_grid_cell[grid_idx] > 0:
                    d[iteration, grid_idx] /= cells_per_grid_cell[grid_idx]

        d /= self.params['t_iteration'] / 1000.
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(d)#, cmap='binary')

        if self.params['training']:
            testtraining = 'training'
        else:
            testtraining = 'testing'

        ax.set_title('Output spikes during %s clustered by x-pos' % testtraining)
        ax.set_ylim((0, d.shape[0]))
        ax.set_xlim((0, d.shape[1]))
        ax.set_ylabel('Iteration')
        ax.set_xlabel('x-pos')
        cbar = pylab.colorbar(cax)
        cbar.set_label('Output rate [Hz]')

        xlabels = ['%.1f' % (float(xtick) / self.n_bins_x) for xtick in self.x_ticks]
        ax.set_xticks(self.x_ticks)
        ax.set_xticklabels(xlabels)
        output_fn = self.params['data_folder'] + 'mpn_output_activity.dat'
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, d)

        output_fig = self.params['figures_folder'] + 'mpn_output_activity.png'
        print 'Saving figure to:', output_fig
        pylab.savefig(output_fig)


    def plot_retinal_displacement(self, stim_range=None, ax=None, lw=3, c='b'):
        if stim_range == None:
            stim_range = self.params['test_stim_range']
        print 'plot_retinal_displacement loads:', self.params['motion_params_fn']
        d = np.loadtxt(self.params['motion_params_fn'])
        it_min = stim_range[0] * self.params['n_iterations_per_stim']
        it_max = stim_range[1] * self.params['n_iterations_per_stim']
        t_axis = d[it_min:it_max, 4]
        t_axis += .5 * self.params['t_iteration']
        x_displacement = np.abs(d[it_min:it_max, 0] - .5)
#        x_displacement = np.zeros(it_max - it_min)
        output_fn = self.params['data_folder'] + 'mpn_xdisplacement.dat'
        print 'Saving data to:', output_fn
        np.savetxt(output_fn, d)

        if ax == None:
            fig = pylab.figure()
            ax = fig.add_subplot(111)

        for stim in xrange(stim_range[0], stim_range[1]):
            it_start_stim = stim * self.params['n_iterations_per_stim']
            it_stop_stim = (stim + 1) * self.params['n_iterations_per_stim'] - 1
            x_displacement_stim = np.abs(d[it_start_stim:it_stop_stim, 0] - .5)
#            x_displacement[it_start_stim:it_stop_stim] = x_displacement_stim
            for it_ in xrange(self.params['n_iterations_per_stim'] - 2):
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
        self.plot_vertical_lines(ax, show_iterations=False)
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
        for i_stim in xrange(params['n_stim']):
            t0 = i_stim * params['n_iterations_per_stim'] * t_scale
            ax.plot((t0, t0), (ymin, ymax), ls='-', lw=2, c='k')
            if show_iterations:
                for it_ in xrange(params['n_iterations_per_stim']):
                    t0 = it_ * t_scale + i_stim * params['n_iterations_per_stim'] * t_scale
                    ax.plot((t0, t0), (ymin, ymax), ls='-.', c='k')
                    ax.annotate(str(it_cnt), xy=(t0 + .5 * t_scale, ymin + (ymax - ymin) * .05))
                    it_cnt += 1


    def plot_raster_sorted(self, title='', cell_type='exc', sort_idx=0):
        """
        sort_idx : the index in tuning properties after which the cell gids are to be sorted for  the rasterplot
        """
        if cell_type == 'exc':
            tp = self.tuning_prop_exc
        else:
            tp = self.tuning_prop_ing

        tp_idx_sorted = tp[:, sort_idx].argsort() # + 1 because nest indexing

        merged_spike_fn = self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn_merged']
        spikes_unsrtd = np.loadtxt(merged_spike_fn)

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
        if sort_idx == 0:
            ax.plot((xlim[0], xlim[1]), (.5, .5), ls='--', lw=3, c='k')
        elif sort_idx == 2:
            ax.plot((xlim[0], xlim[1]), (.0, .0), ls='--', lw=3, c='k')
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

        if len(argv) == 1:
            print '\nPlotting only stim 1!\n\n'
            network_params = simulation_parameters.global_parameters()  
            params = network_params.params
            print '\nPlotting the default parameters give in simulation_parameters.py\n'
            self.run_single_folder_analysis(params, stim_range)
        elif len(argv) == 2:
            folder_name = argv[1]
            params = utils.load_params(folder_name)
            self.run_single_folder_analysis(params, stim_range)
        elif len(argv) == 3:
            if argv[1].isdigit() and argv[2].isdigit():
                stim_range = (int(argv[1]), int(argv[2]))
                network_params = simulation_parameters.global_parameters()  
                params = network_params.params
                print '\nPlotting the default parameters give in simulation_parameters.py\n'
                self.run_single_folder_analysis(params, stim_range)
            else:
                self.run_analysis_for_folders(argv[1:], training_params=training_params)
        elif len(argv) == 4:
            folder_name = argv[1]
            if argv[2].isdigit() and argv[3].isdigit():
                stim_range = (int(argv[2]), int(argv[3]))
                params = utils.load_params(folder_name)
                self.run_single_folder_analysis(params, stim_range)
            else:
                self.run_analysis_for_folders(argv[1:], training_params=training_params)
        elif len(argv) > 4:
            # do the same operation for many folders
            self.run_analysis_for_folders(argv[1:], training_params=training_params)


    def run_single_folder_analysis(self, params, stim_range):
        Plotter = ActivityPlotter(params)#, it_max=1)
#        fig = Plotter.plot_training_sequence()
#        output_fn = params['figures_folder'] + 'training_sequence.png'
#        print 'Saving to', output_fn
#        fig.savefig(output_fn)

    #    Plotter.plot_input()
    #    Plotter.plot_output()
    #    if params['training']:
        (t_axis, x_displacement) = Plotter.plot_retinal_displacement(stim_range=stim_range)
        return (t_axis, x_displacement)

#        del Plotter
        # plot x - pos sorting
    #    fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by x-position', sort_idx=0)
    #    if params['debug_mpn']:
    #        Plotter.plot_input_spikes_sorted(ax, sort_idx=0)
    #    output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_xpos.png'
    #    print 'Saving to', output_fn
    #    fig.savefig(output_fn)

        # plot vx - sorting
    #    fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by preferred speed', sort_idx=2)
    #    if params['debug_mpn']:
    #        Plotter.plot_input_spikes_sorted(ax, sort_idx=2)
    #    output_fn = params['figures_folder'] + 'rasterplot_mpn_in_and_out_vx.png'
    #    print 'Saving to', output_fn
    #    fig.savefig(output_fn)


    def run_analysis_for_folders(self, folders, training_params=None):
        """
        folders -- list of folders with same time/data parameters (wrt to data size to be analysed)


        """
        stim_range = (0, 10) # to be changed

        n_stim = stim_range[1] - stim_range[0]
        # load the first parameter set to determine data size
        params = utils.load_params(folders[0])
        n_idx_per_stim = params['n_iterations_per_stim'] * n_stim
        n_folders = len(folders)
        all_data = np.zeros((n_folders, n_idx_per_stim))

        for i_, folder in enumerate(folders):
            params = utils.load_params(folder)
            (x_data, y_data) = self.run_single_folder_analysis(params, stim_range)
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

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged'])

    training_folder = 'Training_Cluster_taup90000_nStim400_nExcMpn2400_nStates20_nActions21_it15-90000_wMPN-BG1.50_bias10.00/'
    MAC = MetaAnalysisClass(sys.argv, plot_training_folder=training_folder)
#    pylab.show()
