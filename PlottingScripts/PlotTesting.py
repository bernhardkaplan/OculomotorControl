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
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.gridspec as gridspec
import MergeSpikefiles
import FigureCreator
import json
from MetaAnalysisClass import MetaAnalysisClass
import PlotMPNActivity

class PlotTesting(MetaAnalysisClass):

    def __init__(self, argv, verbose=False):
        self.verbose = verbose
        self.rp_markersize = 2
        self.tick_interval = 8



        plot_params = {'backend': 'png',
                      'axes.labelsize': 22,
                      'axes.titlesize': 24,
                      'text.fontsize': 20,
                      'xtick.labelsize': 16,
                      'ytick.labelsize': 16,
                      'legend.pad': 0.2,     # empty space around the legend box
                      'legend.fontsize': 20,
                       'lines.markersize': 1,
                       'lines.markeredgewidth': 0.,
                       'lines.linewidth': 1,
                      'font.size': 12,
                      'path.simplify': False,

                      'figure.subplot.left':.20,
                      'figure.subplot.bottom':.07,
                      'figure.subplot.right':.94,
                      'figure.subplot.top':.95,
                      'figure.subplot.hspace':.04, 
                      'figure.subplot.wspace':.32}

#                      'figure.subplot.left':.12,
#                      'figure.subplot.bottom':.12,
#                      'figure.subplot.right':.94,
#                      'figure.subplot.top':.97,
#                      'figure.subplot.hspace':.06, 
#                      'figure.subplot.wspace':.32}

        #              'figure.figsize': get_fig_size(800)}
        plt.rcParams.update(plot_params)

        MetaAnalysisClass.__init__(self, argv, verbose) # call the constructor of the super/mother class
        # the constructer of the MetaAnalysisClass calls run_super_plot
        # with params and stim_range retrieved from the command line arguments

    def plot_mpn_spikes(self):
        P = PlotMPNActivity.ActivityPlotter(self.params)
        title = 'Exc cells sorted by x-position'
        fig, ax = P.plot_raster_sorted(title=title, sort_idx=0, t_range=self.t_range)


    def run_super_plot(self, params, stim_range):


        self.params = params
        utils.merge_spikes(params)


        self.tuning_prop = np.loadtxt(self.params['tuning_prop_exc_fn'])
        if stim_range == None:
#            self.stim_range = (self.params['test_stim_range'], self.params['test_stim_range'][-1] + 1)
            self.stim_range = range(len(self.params['test_stim_range']))
#            self.stim_range = self.params['test_stim_range']
        else:
            self.stim_range = range(stim_range[0], stim_range[1])

#        if len(self.stim_range) == 0: # only one test stimulus
#            self.stim_range = (stim_range[0]

        print 'debug self.stim_range:', self.stim_range
        self.n_stim = self.stim_range[-1] - self.stim_range[0] + 1
        self.it_range = [0, 0]
        self.it_range[0] = self.stim_range[0] * self.params['n_iterations_per_stim']
        self.it_range[1] = (self.stim_range[-1] + 1) * self.params['n_iterations_per_stim']

        self.t_range = [0, 0]
        self.t_range[0] = self.stim_range[0] * self.params['t_iteration'] * self.params['n_iterations_per_stim']
        self.t_range[1] = (self.stim_range[-1] + 1) * self.params['t_iteration'] * self.params['n_iterations_per_stim']

#        self.plot_mpn_spikes()
        print 'Plotting stim_range', self.stim_range
        print 'Plotting t_range', self.t_range
        self.plot_retinal_displacement()


    def plot_network_readout_as_colormap(self, tp_idx=0):
        spike_fn = self.params['spiketimes_folder'] + self.params['mpn_exc_spikes_fn_merged']
        print 'Loading MPN spikes from:', spike_fn
        d_spikes = np.loadtxt(spike_fn)
        n_time = 1
        n_bins = 50
        cmap_data = np.zeros((n_time * (self.it_range[1] - self.it_range[0]), n_bins))

        vec_avg = np.zeros(n_time * (self.it_range[1] - self.it_range[0]))
        y_axis = np.linspace(np.min(self.tuning_prop[:, tp_idx]), np.max(self.tuning_prop[:, tp_idx]), n_bins)
        for it_ in xrange(self.it_range[1] - self.it_range[0]):
            t0, t1 = it_ * self.params['t_iteration'], (it_ + 1) * self.params['t_iteration']
            vad = utils.vector_average_spike_data(d_spikes, self.tuning_prop, n_bins, t_range=(t0, t1), n_time=n_time, tp_idx=tp_idx)
            cmap_data[it_ * n_time : (it_ + 1) * n_time, :] = vad
            for i_time in xrange(n_time):
                vec_avg[it_ * n_time + i_time] = np.sum(vad[i_time, :] * y_axis)
                
        ################################
        # PLOT 0: COLORMAP
        ################################
        cmap_data = cmap_data.transpose()
        ax = self.plot_matrix(cmap_data)
        xticks = ax.get_xticks() 
        new_xticklabels = []
        for i_, xtick in enumerate(xticks):
            new_xticklabels.append('%d' % (xtick * self.params['t_iteration'] / float(n_time)))
        ax.set_xticklabels(new_xticklabels)
        n_y_ticks = 5
        ax.set_yticks(np.linspace(0, n_bins - 1, n_y_ticks, endpoint=True))
        yticks = ax.get_yticks() 
        new_yticklabels = []
        for i_, ytick in enumerate(yticks):
            new_yticklabels.append('%.1f' % (y_axis[np.int(ytick)]))
        ax.set_yticklabels(new_yticklabels)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot((xlim[0], xlim[1]), (.5 * n_bins, .5 * n_bins), ls='--', c='w', lw=3)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Retinal displacement')
        output_fn = self.params['figures_folder'] + 'retinal_displacement_cmap_%d-%d.png' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving colormap figure to:', output_fn
        plt.savefig(output_fn, dpi=300)
        output_fn = self.params['data_folder'] + 'retinal_displacement_cmap_%d-%d.dat' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving colormap data to:', output_fn
        np.savetxt(output_fn, cmap_data)



    def plot_retinal_displacement(self):
#        self.plot_network_readout_as_colormap()

        ################################
        # PLOT 1: Stimulus trajectory: single trials and average
        ################################
        actions_taken = np.loadtxt(self.params['actions_taken_fn'])
        stim_params = np.loadtxt(self.params['testing_stimuli_fn'])
        trajectory = np.zeros(self.params['n_iterations'] + 1)
        cnt_ = 0
        all_displ = np.zeros((self.params['n_iterations_per_stim'], self.n_stim))
        avg_displ = np.zeros((self.params['n_iterations_per_stim'], 2))
        avg_displ_good = np.zeros((self.params['n_iterations_per_stim'], 2)) # the avg only of the good stimuli
        problematic_stimuli_idx = [] 
        problematic_stimuli = [] # stim params
        good_stimuli_idx = []
        good_stimuli = [] 

        for i_stim in xrange(len(self.stim_range)):
            trajectory[cnt_] = stim_params[i_stim, 0]
            for it_ in xrange(self.params['n_iterations_per_stim']):
                trajectory[cnt_ + 1] = trajectory[cnt_] + (stim_params[i_stim, 2] - actions_taken[cnt_ + 1, 0]) * self.params['t_iteration'] / 1000.
                #all_displ[it_, i_stim] = np.abs(trajectory[cnt_ + 1] - .5)
                all_displ[it_, i_stim] = np.abs(trajectory[cnt_] - .5)
                cnt_ += 1
            # distinguish if stimulus is 'problematic' or if it is OK
            i1 = (i_stim + 1) * self.params['n_iterations_per_stim']
            mean_displ_end = np.abs(trajectory[i1-12:i1-2].mean() - .5)
            if mean_displ_end > .15:
                problematic_stimuli_idx.append(i_stim)
            else:
                good_stimuli_idx.append(i_stim)
            
        print 'Problematic_stimuli_idx:', problematic_stimuli_idx
        print 'good_stimuli_idx:', good_stimuli_idx
        good_stimuli_idx = np.array(good_stimuli_idx)

        for it_ in xrange(self.params['n_iterations_per_stim']):
            avg_displ[it_, 0] = all_displ[it_, :].mean()
            avg_displ[it_, 1] = all_displ[it_, :].std()
            avg_displ_good[it_, 0] = all_displ[it_, np.array(good_stimuli_idx)].mean()
            avg_displ_good[it_, 1] = all_displ[it_, np.array(good_stimuli_idx)].std()

        avg_displ[:, 1] /= np.sqrt(self.params['n_stim'])
        avg_displ_good[:, 1] /= np.sqrt(len(good_stimuli_idx))
        fig0 = plt.figure(figsize=FigureCreator.get_fig_size(800, portrait=True))
        ax0 = fig0.add_subplot(111)
        fig = plt.figure(figsize=FigureCreator.get_fig_size(800, portrait=True))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_title('Test performance')
#        ax0 = fig.add_subplot(311)
#        ax1 = fig.add_subplot(312)
#        ax2 = fig.add_subplot(313)

        clim = (min(np.abs(stim_params[:, 2])), max(np.abs(stim_params[:, 2])))
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])#, clip=True)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
        m.set_array(np.arange(clim[0], clim[1], 0.01))
        v_colors = m.to_rgba(np.abs(stim_params[:, 2]))

#        for i_stim in xrange(len(self.params['test_stim_range'])):
        for i_stim in xrange(len(self.stim_range)):
            problematic_stimulus = False
            i0 = i_stim * self.params['n_iterations_per_stim']
            i1 = (i_stim + 1) * self.params['n_iterations_per_stim']
            mean_displ_end = np.abs(trajectory[i1-12:i1-2].mean() - .5)

            lc = v_colors[i_stim]

            if mean_displ_end > .15:
#            if np.where(trajectory[i0:i1] > 1.0)[0].size > 0 or np.where(trajectory[i0:i1] < 0.)[0].size > 0:
#                lc = 'k'
                print 'Problematic Stim id ', i_stim, stim_params[i_stim, :]
                problematic_stimuli.append(stim_params[i_stim, :])
#                problematic_stimuli_idx.append(i_stim)
                problematic_stimulus = True
                #print 'debug bad', np.abs(trajectory[i0+4:i1-2].mean() - .5)
            else:
                problematic_stimulus = False
#                lc = 'k'
                #print 'debug good', np.abs(trajectory[i0+4:i1-2].mean() - .5)
                good_stimuli.append(stim_params[i_stim, :])
            ax1.plot(range(self.params['n_iterations_per_stim']), trajectory[i0:i1], color=lc, lw=1, alpha=0.6)
            ax0.plot(range(i0, i1), trajectory[i0:i1], color=lc, lw=1)
            #ax.plot(range(trajectory.size), trajectory)

        # restrict the x-range for some example stimuli
        ax0_stim_range = (0, 100)
        ax0_xlim = (ax0_stim_range[0] * self.params['n_iterations_per_stim'], ax0_stim_range[1] * self.params['n_iterations_per_stim'])
        ax0.set_xlim((ax0_xlim[0], ax0_xlim[1]))
        xticks0 = ax0.get_xticks() 
        new_xticklabels = []
        for i_, xtick in enumerate(xticks0):
            new_xticklabels.append('%d' % (xtick * self.params['t_iteration']))
        ax0.set_xticklabels(new_xticklabels)
        ax0.set_title('Test performance: single trials and average')

        new_xticklabels = []
        ax1.set_xticklabels(new_xticklabels)
#        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Retinal\ndisplacement')
        ax1.set_xlim((0., self.params['n_iterations_per_stim']))# * self.params['t_iteration']))
        xlim = ax1.get_xlim()
        ax1.plot((xlim[0], xlim[1]), (.5, .5), ls='--', c='k', lw=4)
        ax1.plot((xlim[0], xlim[1]), (.5 + self.params['blur_X'], .5 + self.params['blur_X']), ls='-.', c='k', lw=2)
        ax1.plot((xlim[0], xlim[1]), (.5 - self.params['blur_X'], .5 - self.params['blur_X']), ls='-.', c='k', lw=2)
        ax1.set_ylim((0., 1.))
#        cb = fig.colorbar(m, ax=ax1)
#        cb.set_label('$|v_{stim}|$')#, fontsize=24)

#        ax0.set_xticklabels(new_xticklabels)
        ax0.set_ylabel('Stimulus position')
        xlim0 = ax0.get_xlim()
        ax0.plot((xlim0[0], xlim0[1]), (.5, .5), ls='--', c='k', lw=3, label='Stimulus radius $\\beta_X$')
        ax0.set_ylim((0.1, 0.9))
#        ax0.set_ylim((-0.1, 1.1))
        ax0.set_xlabel('Time [ms]')

        plt.legend()

#        plt.errorbar(range(self.params['n_iterations_per_stim']), avg_displ[:, 0], yerr=avg_displ[:, 1], c='g', lw=4)
#        avg_label = '$\\frac{1}{N_{test}} \sum_{i=1}^{N_{test}} |x^{stim}_{i}(t) - 0.5|$'
        avg_label = '$\\frac{1}{N_{test}} \sum_{i=1}^{N_{test}} |x^{stim}_{i}(t) - 0.5|$'
        plt.errorbar(range(self.params['n_iterations_per_stim']), avg_displ_good[:, 0], yerr=avg_displ_good[:, 1], c='b', lw=3, label=avg_label)
        ax2.set_xlim(xlim) # set the same xlim as above, as these two axes share the x-ticks on the x-axis
        ylim = ax2.get_ylim()
        ax2.set_ylim((0., 0.3))
#        ax2.set_ylim((0., ylim[1]))
#        ax2.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=3)
        ax2.plot((xlim[0], xlim[1]), (self.params['blur_X'], self.params['blur_X']), ls='-.', c='k', lw=2, label='Stimulus radius $\\beta_X$')
        plt.legend()

        xticks = ax2.get_xticks() 
        new_xticklabels = []
        for i_, xtick in enumerate(xticks):
            new_xticklabels.append('%d' % (xtick * self.params['t_iteration']))
        ax2.set_xticklabels(new_xticklabels)
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Average absolute\nretinal displacement')

        # SAVING
        output_fn = self.params['figures_folder'] + 'retinal_displacement_avg_%d-%d.png' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving figures to:', output_fn
        plt.savefig(output_fn, dpi=200)
        output_fn = self.params['data_folder'] + 'retinal_displacement_all_trj_%d-%d.dat' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving all data (single runs) to:', output_fn
        np.savetxt(output_fn, all_displ)
        output_fn = self.params['data_folder'] + 'retinal_displacement_avg_displ_%d-%d.dat' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving avg displacement data to:', output_fn
        np.savetxt(output_fn, avg_displ)

        output_fn = self.params['data_folder'] + 'problematic_stimuli_idx.txt'
        np.savetxt(output_fn, np.array(problematic_stimuli_idx, dtype=np.int))
        output_fn = self.params['data_folder'] + 'problematic_stimuli.txt'
        print 'Saving the problematic stimuli to:', output_fn
        problematic_stimuli = np.array(problematic_stimuli)
        good_stimuli = np.array(good_stimuli)
        np.savetxt(output_fn, problematic_stimuli)
        print 'Problematic stimuli:', problematic_stimuli
#        print 'Good stimuli:', good_stimuli
        ax = None
        if len(problematic_stimuli) > 0:
            ax = self.plot_stimuli_scatter(problematic_stimuli, 'r', '^', ax, label_txt='Problematic stimuli')
        if len(good_stimuli) > 0:
            ax = self.plot_stimuli_scatter(good_stimuli, 'b', 'o', ax, label_txt='Good stimuli')
        output_fn = self.params['figures_folder'] + 'good_and_problematic_stimuli_scatter.png'
        print 'Saving stimuli scatter plot to:', output_fn
        plt.savefig(output_fn, dpi=200)

    def plot_stimuli_scatter(self, stim_params, color='b', marker='o', ax=None, label_txt=''):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(stim_params[:, 0], stim_params[:, 2], color=color, marker=marker, linestyle='', markersize=10, label=label_txt)
        ax.set_xlabel('$x_{stim}$')
        ax.set_ylabel('$v_{stim}$')
        if label_txt != '':
            ax.legend()
        return ax

        """
        ################################
        # PLOT 1: Vector average, single trials
        ################################
        # VEC - AVG plot
        fig = plt.figure()
        ax = fig.add_subplot(211)
        n_stim = self.stim_range[-1] + 1 - self.stim_range[0]
        all_va = np.zeros((n_stim, n_time * self.params['n_iterations_per_stim']))
        for i_stim in xrange(self.stim_range[-1] + 1- self.stim_range[0]):
            idx0 = i_stim * self.params['n_iterations_per_stim'] * n_time
            idx1 = (i_stim + 1) * self.params['n_iterations_per_stim'] * n_time
            y_data = vec_avg[idx0:idx1]
            all_va[i_stim, :] = np.abs(y_data - .5)
            ax.plot(range(n_time * self.params['n_iterations_per_stim']), y_data, c='k', lw=1)

        xlim = ax.get_xlim()
        ax.plot((xlim[0], xlim[1]), (.5, .5), ls='--', c='k', lw=3)

        # no xticks, because they are the same as ax2
        new_xticklabels = []
        for i_, xtick in enumerate(xticks):
            new_xticklabels.append('')
#            new_xticklabels.append('%d' % (xtick * self.params['t_iteration'] / float(n_time)))
        ax.set_xticklabels(new_xticklabels)
        ax.set_ylabel('Retinal\ndisplacement')

        ################################
        # PLOT 2: Average displacement
        ################################
        va_avg = np.zeros((n_time * self.params['n_iterations_per_stim'], 2))
        for i_time in xrange(n_time * self.params['n_iterations_per_stim']):
            va_avg[i_time, 0] = all_va[:, i_time].mean()
            va_avg[i_time, 1] = all_va[:, i_time].std()
        va_avg[:, 1] /= np.sqrt(n_stim)
        ax2 = fig.add_subplot(212)
        plt.errorbar(range(n_time * self.params['n_iterations_per_stim']), va_avg[:, 0], yerr=va_avg[:, 1], c='b', lw=5)
        ax2.set_xlim(xlim) # set the same xlim as above, as these two axes share the x-ticks on the x-axis
#        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        ax2.set_ylim((0., ylim[1]))
        ax2.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=3)

        xticks = ax2.get_xticks() 
        new_xticklabels = []
        for i_, xtick in enumerate(xticks):
            new_xticklabels.append('%d' % (xtick * self.params['t_iteration'] / float(n_time)))
        ax2.set_xticklabels(new_xticklabels)
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Average absolute\nretinal displacement')

        # SAVING
        output_fn = self.params['figures_folder'] + 'retinal_displacement_avg_%d-%d.png' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving figures to:', output_fn
        plt.savefig(output_fn, dpi=300)
        output_fn = self.params['data_folder'] + 'retinal_displacement_allVA_%d-%d.dat' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving colormap data to:', output_fn
        np.savetxt(output_fn, all_va)
        output_fn = self.params['data_folder'] + 'retinal_displacement_avg_%d-%d.dat' % (self.stim_range[0], self.stim_range[-1])
        print 'Saving colormap data to:', output_fn
        np.savetxt(output_fn, va_avg)
        """


    def plot_matrix(self, d, title=None, clim=None):

#        cmap_name = 'bwr'
        cmap_name = None
        if clim != None:
            norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])#, clip=True)
            m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
            m.set_array(np.arange(clim[0], clim[1], 0.01))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print "plotting .... "
        if clim != None:
            cax = ax.pcolormesh(d, cmap=cmap_name, vmin=clim[0], vmax=clim[1])
        else:
            cax = ax.pcolormesh(d, cmap=cmap_name)

        ax.set_ylim((0, d.shape[0]))
        ax.set_xlim((0, d.shape[1]))
        if title != None:
            ax.set_title(title)
        cbar = plt.colorbar(cax)
        cbar.set_label('Confidence')
        return ax


if __name__ == '__main__':

#    MAC = MetaAnalysisClass(sys.argv)
    P = PlotTesting(sys.argv, verbose=True)
    plt.show()

