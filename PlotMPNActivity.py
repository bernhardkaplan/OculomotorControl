import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import sys
#import rcParams
#rcP= rcParams.rcParams
import os
import utils


class ActivityPlotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        if it_max == None:
            self.it_max = self.params['n_iterations']
        else:
            self.it_max = it_max

        self.n_bins_x = 20
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
            fn_to_match = (self.params['input_nspikes_fn_mpn'] + 'it%d' % iteration).rsplit('/')[-1]
            list_of_files = utils.find_files(self.params['input_folder_mpn'], fn_to_match)
            for fn_ in list_of_files:
                fn = self.params['input_folder_mpn'] + fn_
#                print 'Loading:', fn
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

        ax.set_title('Input spikes')
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
        print 'Saving figure to:', output_fn
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
        merged_spike_fn = self.params['spiketimes_folder_mpn'] + self.params['merged_exc_spikes_fn_mpn']
        utils.merge_and_sort_files(self.params['spiketimes_folder_mpn'] + self.params['exc_spikes_fn_mpn'], merged_spike_fn)
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

        ax.set_title('Output spikes clustered by x-pos')
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
        print 'Saving figure to:', output_fn
        pylab.savefig(output_fig)










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


    Plotter = ActivityPlotter(params)#, it_max=1)
    Plotter.plot_input()
    Plotter.plot_output()
    pylab.show()
