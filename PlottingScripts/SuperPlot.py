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

class MetaAnalysisClass(object):
    def __init__(self, argv, verbose=False):
        stim_range = None
        if len(sys.argv) == 1:
            if verbose:
                print 'Case 1', sys.argv
            network_params = simulation_parameters.global_parameters()  
            params = network_params.params
            self.run_super_plot(params, stim_range)

        elif len(sys.argv) == 2:                                # plot_ [FOLDER]
            if verbose:
                print 'Case 2', sys.argv
            folder_name = sys.argv[1]
            params = utils.load_params(folder_name)
            self.run_super_plot(params, stim_range)
        elif len(sys.argv) == 3: 
            if verbose:
                print 'Case 3', sys.argv
            if sys.argv[1].isdigit() and sys.argv[2].isdigit(): #  plot_ [STIM_1] [STIM_2]
                stim_range = (int(sys.argv[1]), int(sys.argv[2]))
                network_params = simulation_parameters.global_parameters()  
                params = network_params.params
                self.run_super_plot(params, stim_range)
            else:
                for fn in sys.argv[1:]:                         # plot_ [FOLDER] [FOLDER]
                    params = utils.load_params(fn)
                    self.run_super_plot(params, stim_range)
        elif len(sys.argv) == 4:                                
            if verbose:
                print 'Case 4', sys.argv
            folder_name = sys.argv[1]
            if sys.argv[2].isdigit() and sys.argv[3].isdigit(): # plot_ [FOLDER] [STIM_1] [STIM_2]
                stim_range = (int(sys.argv[2]), int(sys.argv[3]))
                params = utils.load_params(folder_name)
                # create one figure for the full stim range
                self.run_super_plot(params, stim_range)

                # create separate figures for each individual stimulus
#                for i_stim in xrange(stim_range[0], stim_range[1]):
#                    self.run_super_plot(params, (i_stim, i_stim + 1))
            else:
                # create separate figures for each individual folder
                for fn in sys.argv[1:]:                         # plot_ [FOLDER_1] [FOLDER_2] [FOLDER_3]
                    params = utils.load_params(fn)
                    self.run_super_plot(params, stim_range)
        elif len(sys.argv) > 4:                                 # plot_ [FOLDER_1] [FOLDER_2] .... [FOLDER_N]
            # create separate figures for each individual folder
            if verbose:
                print 'Case 5', sys.argv
            for fn in sys.argv[1:]:
                params = utils.load_params(fn)
                self.run_super_plot(params, stim_range)



    def run_super_plot(self, params, stim_range):
        """
        if stim_range == None:
            plot full range of iterations
        """
        raise NotImplementedError, 'To be implemented by sub class'
            


class PlotEverything(MetaAnalysisClass):

    def __init__(self, argv, verbose=False):
        self.verbose = verbose
        self.rp_markersize = 2
        MetaAnalysisClass.__init__(self, argv, verbose) # call the constructor of the super/mother class


    def run_super_plot(self, params, stim_range):
        self.params = params

        utils.merge_spikes(params)
        print 'run_super_plot: folder %s, stim_range' % params['folder_name'], stim_range

        if stim_range == None:
            stim_range = range(params['n_stim'])

        t_range = [0, 0]
        t_range[0] = stim_range[0] * self.params['t_iteration'] * self.params['n_iterations_per_stim']
        t_range[1] = stim_range[1] * self.params['t_iteration'] * self.params['n_iterations_per_stim']

        print 'xlim:', t_range

        figsize = FigureCreator.get_fig_size(800, portrait=True)
        self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(4, 1, height_ratios=(2, 2, 1, 1))

#        self.plot_bg_spikes(t_range)
#        self.plot_mpn_spikes(t_range)
        self.plot_retinal_displacement_and_reward(stim_range)



    def plot_bg_spikes(self, t_range):

        ax0 = plt.subplot(self.gs[0])

        marker = 'o'
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for z_, cell_type in enumerate(['d1', 'd2', 'actions']):

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

        for i_, gid in enumerate(tp_idx_sorted):
            spikes = utils.get_spiketimes(spikes_unsrtd, gid + 1)
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * self.tuning_prop_exc[gid, sort_idx]
            ax1.plot(spikes, y_, 'o', markersize=self.rp_markersize, color='k')

        if t_range != None:
            ax1.set_xlim(t_range)

        if sort_idx == 0:
            ax1.set_ylim((0., 1.))
        if sort_idx == 1:
            ax1.set_ylim((np.min(self.tuning_prop_exc[:, 2]), np.max(self.tuning_prop_exc[:, 2])))

    def plot_retinal_displacement_and_reward(self, stim_range):
        print 'stim_range', stim_range
        if self.params['training']:
            fn = self.params['motion_params_training_fn']
        else:
            fn = self.params['motion_params_testing_fn']
        mp = np.loadtxt(fn)
        actions = np.loadtxt(self.params['actions_taken_fn'])
        rewards = np.loadtxt(self.params['rewards_given_fn'])
        K_vec = np.loadtxt(self.params['K_values_fn']) 
        ax2 = plt.subplot(self.gs[2])
        ax3 = plt.subplot(self.gs[3])
        color = 'k'
        for i_stim in stim_range:
            t0 = i_stim * self.params['t_iteration'] * self.params['n_iterations_per_stim'] + 1 * self.params['t_iteration'] # + 1 because stimulus appears in iteration 1 (not 0) within a stimulus
            t1 = i_stim * self.params['t_iteration'] * self.params['n_iterations_per_stim'] + 2 * self.params['t_iteration'] # + 2 for the consequence of the action
            x_stim = mp[i_stim, 0]
            v_eye = actions[i_stim, 0]
            x_after = utils.get_next_stim(self.params, mp[i_stim, :], actions[i_stim, 0])[0]
            print t0, t1, x_stim, x_after, v_eye, mp[i_stim, :]
            ax2.plot(np.array([t0, t1]), np.array([x_stim, x_after]), color=color, lw=3, marker='o')


        for i_stim in stim_range:
            idx0 = i_stim * self.params['n_iterations_per_stim']
            idx1 = (i_stim + 1) * self.params['n_iterations_per_stim']
            ax3.plot(range(idx0, idx1), K_vec[idx0:idx1])
#            t0 = idx0 * self.params['t_iteration']

#            for it_ in xrange(0, self.params['n_iterations_per_stim']):
#                it_idx = i_stim * self.params['n_iterations_per_stim'] + it_
#                print 'debug', K_vec[it_idx], rewards[i_stim], it_idx, i_stim
#                assert (K_vec[it_idx] == rewards[i_stim]), 'ERROR in plot_retinal_displacement_and_reward: Mismatch between K_vec and rewards'
#                t0 = (i_stim * self.params['n_iterations_per_stim'] + it_) * self.params['t_iteration']
#                t1 = (i_stim * self.params['n_iterations_per_stim'] + it_ + 1) * self.params['t_iteration']
#                ax3.plot([t0, t1], [K_vec[it_idx], K_vec[it_idx]])



if __name__ == '__main__':

#    MAC = MetaAnalysisClass(sys.argv)
    P = PlotEverything(sys.argv, verbose=True)
    plt.show()
