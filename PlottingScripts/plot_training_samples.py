import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import matplotlib
from matplotlib import mlab, cm
import sys
import os
import utils
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from FigureCreator import plot_params
import VisualInput
import BasalGanglia
import FigureCreator as FC

plot_params['figure.subplot.right'] = 0.95

pylab.rcParams.update(plot_params)

class Plotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        tp_fn = self.params['tuning_prop_exc_fn']
        print 'Loading', tp_fn
        self.tp = np.loadtxt(tp_fn)
        print 'Loading', self.params['receptive_fields_exc_fn']
        self.rfs = np.loadtxt(self.params['receptive_fields_exc_fn'])
        self.mp_training = None


    def plot_training_sample_space(self, plot_process=False, motion_params_fn=None, n_samples_to_plot=None):
        if plot_process:
            try:
                if motion_params_fn == None:
                    training_stim_fn = self.params['motion_params_fn']
                print 'Loading training stimuli data from:', motion_params_fn
                d = np.loadtxt(motion_params_fn)
            except:
                if motion_params_fn == None:
                    motion_params_fn = self.params['motion_params_precomputed_fn']
                print 'Loading training stimuli data from:', motion_params_fn
                d = np.loadtxt(motion_params_fn)
        else:
            if motion_params_fn == None:
                motion_params_fn = self.params['training_stimuli_fn']
            print 'Loading training stimuli data from:', motion_params_fn 
            d = np.loadtxt(motion_params_fn)
        self.mp_training = d

        fig = pylab.figure()#figsize=(12, 12))
        ax1 = fig.add_subplot(111)

        patches = []

        for gid in xrange(self.params['n_exc_mpn']):
            ax1.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c='k', markersize=2)
            ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2], linewidth=0, alpha=0.1)
            ellipse.set_facecolor('b')
            patches.append(ellipse)
            ax1.add_artist(ellipse)

        # plot the stimulus start points
#        for i_ in xrange(self.params['n_stim']):

        ylim = ax1.get_ylim()
        ax1.plot((.5, .5), (ylim[0], ylim[1]), ls='--', c='k', lw=3)

        if n_samples_to_plot == None:
            n_samples_to_plot = d[:, 0].size

        for i_ in xrange(n_samples_to_plot):
            if plot_process:
#                idx = i_ * self.params['n_iterations_per_stim']
#                mp = d[idx, :]
                mp = d[i_, :]
                ax1.plot(mp[0], mp[2], '*', markersize=10, color='y', markeredgewidth=1)#, zorder=100)
                idx_stop = (i_ + 1) * self.params['n_iterations_per_stim']
                mps = d[idx:idx_stop, :]
                ax1.plot(mps[:, 0], mps[:, 2], '--', color='r', lw=3)
                ellipse = mpatches.Ellipse((mp[0], mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0)
                ellipse.set_facecolor('r')
                patches.append(ellipse)
                ax1.add_artist(ellipse)
            else:
                mp = d[i_, :]
                ax1.plot(mp[0], mp[2], '*', markersize=10, color='y', markeredgewidth=1)
                ellipse = mpatches.Ellipse((mp[0], mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0, alpha=0.2)
                ellipse.set_facecolor('r')
                patches.append(ellipse)
                ax1.add_artist(ellipse)
        collection = PatchCollection(patches, alpha=0.1)
        ax1.add_collection(collection)

#        ax1.text(.4, ylim[0] + .1 * (ylim[1] - ylim[0]), 'Visual field center', fontsize=16)

        ax1.set_title('Training stimuli state space')
        ax1.set_xlabel('Stimulus position') 
        ax1.set_ylabel('Stimulus speed vx') 
        output_fig = self.params['figures_folder'] + 'stimulus_state_space_%.2f_%.2f_n%d.png' % (self.params['training_stim_noise_x'], self.params['training_stim_noise_v'], n_samples_to_plot)
        print 'Saving to:', output_fig
        pylab.savefig(output_fig, dpi=200)



    def plot_precomputed_actions(self, training_stim_fn=None, plot_cells=True, n_samples_to_plot=None):

        print 'Loading actions taken from:', self.params['actions_taken_fn']


#        actions_taken = np.loadtxt(self.params['actions_taken_fn'])
#        action_indices = actions_taken[:, 2]
        if training_stim_fn == None:
            training_stim_fn = self.params['training_stimuli_fn']
        print 'Loading training stimuli from:', training_stim_fn
        d = np.loadtxt(training_stim_fn)
        VI = VisualInput.VisualInput(self.params)
        BG = BasalGanglia.BasalGanglia(self.params, dummy=True)
        supervisor_states, action_indices, motion_params_precomputed = VI.get_supervisor_actions(d, BG)
        self.mp_training = d

        patches = []
#        fig = pylab.figure(figsize=(10, 10))
        fig = pylab.figure(figsize=FC.get_fig_size(800, portrait=False))
        ax1 = fig.add_subplot(111)

        # define the colormap
        cmap = matplotlib.cm.jet
        # extract all colors from the cmap
        cmaplist = [cmap(i) for i in xrange(cmap.N)]
        # force the first color entry to be grey #cmaplist[0] = (.5,.5,.5,1.0)
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = range(self.params['n_actions'])
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(np.arange(bounds[0], bounds[-1], 1.))
        rgba_colors = m.to_rgba(bounds)
#        cb = fig.colorbar(m)
#        cb.set_label('Action indices')#, fontsize=24)

        if plot_cells:
            for gid in xrange(self.params['n_exc_mpn']):
                ax1.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c='k', markersize=2)
                ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2], linewidth=0, alpha=0.1)
                ellipse.set_facecolor('b')
                patches.append(ellipse)
                ax1.add_artist(ellipse)

        print 'type action_indices', type(action_indices)
        colors = m.to_rgba(action_indices)
#        print 'debug colors', colors, '\n\n', action_indices

        if n_samples_to_plot == None:
            n_samples_to_plot = d[:, 0].size
#        if self.params['reward_based_learning']:
#            for i_ in xrange(len(action_indices)):
#                stim_idx = i_
#                mp = d[stim_idx, :]
#                ax1.plot(mp[0], mp[2], '*', markersize=10, color=colors[i_], markeredgewidth=1)
#                ellipse = mpatches.Ellipse((mp[0], mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0, alpha=0.2)
#                ellipse.set_facecolor('r')
#                patches.append(ellipse)
#                ax1.add_artist(ellipse)

#        else:
        for i_ in xrange(n_samples_to_plot):
            mp = d[i_, :]
            ax1.plot(mp[0], mp[2], '*', markersize=10, color=colors[i_], markeredgewidth=1)
            ax1.text(mp[0] + 0.01, mp[2] + 0.04, '%d' % action_indices[i_], fontsize=12)

            ellipse = mpatches.Ellipse((mp[0], mp[2]), self.params['blur_X'], self.params['blur_V'], linewidth=0, alpha=0.2)
            ellipse.set_facecolor('r')
            patches.append(ellipse)
            ax1.add_artist(ellipse)


#        ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
#        cb = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

#        collection = PatchCollection(patches)
        ylim = ax1.get_ylim()
#        xlim = ax1.get_ylim()
        ax1.plot((.5, .5), (ylim[0], ylim[1]), ls='--', c='k', lw=3)
        ax1.set_ylim((0.8 * ylim[0], 0.8 * ylim[1]))
#        ax1.text(.4, ylim[0] + .1 * (ylim[1] - ylim[0]), 'Visual field center', fontsize=16)
#        ax1.add_collection(collection)
        ax1.set_title('Stimulus and tuning property space')
        ax1.set_xlabel('Stimulus and receptive field position $x$') 
        ax1.set_ylabel('Stimulus and preferred speed $v_x$') 
        output_fig = self.params['figures_folder'] + 'stimulus_state_space_with_precomputed_actions_%.2f_%.2f_n%d.png' % (self.params['training_stim_noise_x'], self.params['training_stim_noise_v'], n_samples_to_plot)
        print 'Saving to:', output_fig
        pylab.savefig(output_fig, dpi=200)
        return ax1


    def plot_training_sample_histograms(self):

        if self.mp_training == None:
            fn = self.params['training_stimuli_fn']
            print 'Loading training stimuli data from:', fn
            self.mp_training = np.loadtxt(fn)

        n_bins = 100
        d = self.mp_training[:, 0]
        cnt_x, bins_x = np.histogram(d, bins=n_bins, range=(np.min(d), np.max(d)))

        d = self.mp_training[:, 2]
        cnt_v, bins_v = np.histogram(d, bins=n_bins, range=(np.min(d), np.max(d)))

        fig = pylab.figure()
        ax1 = fig.add_subplot(211)
        ax1.bar(bins_x[:-1], cnt_x, width=bins_x[1]-bins_x[0])
        ax1.set_xlabel('x-pos training stimuli')
        ax1.set_ylabel('Count')

        print 'Position histogram:'
        print 'First half:', bins_x[:n_bins/2]
        print 'Sum first half:', cnt_x[:n_bins/2].sum()
        print 'Second half:', bins_x[-(n_bins/2 + 1):]
        print 'Sum second half:', cnt_x[-(n_bins/2 + 1):].sum()


        ax2 = fig.add_subplot(212)
        ax2.bar(bins_v[:-1], cnt_v, width=bins_v[1]-bins_v[0])
        ax2.set_xlabel('Speed of training stimuli')
        ax2.set_ylabel('Count')

        print 'Velocity histogram:'
        print 'First half:', bins_v[:n_bins/2]
        print 'Sum first half:', cnt_v[:n_bins/2].sum()
        print 'Second half:', bins_v[-(n_bins/2 + 1):]
        print 'Sum second half:', cnt_v[-(n_bins/2 + 1):].sum()


if __name__ == '__main__':

    if len(sys.argv) > 1:
        try: 
            params = utils.load_params_from_file(sys.argv[1])
        except:
            training_stim_fn = sys.argv[1]
            import simulation_parameters
            param_tool = simulation_parameters.global_parameters()
            params = param_tool.params
    else:
        import simulation_parameters
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params
        training_stim_fn = None

    Plotter = Plotter(params)#, it_max=1)
    N = 30
#    N = None
#    Plotter.plot_training_sample_space(plot_process=False, motion_params_fn=training_stim_fn, n_samples_to_plot=N)
#    Plotter.plot_training_sample_space(plot_process=True)
    Plotter.plot_precomputed_actions(training_stim_fn=training_stim_fn, plot_cells=True, n_samples_to_plot=N)
#    Plotter.plot_training_sample_histograms()

    pylab.show()
