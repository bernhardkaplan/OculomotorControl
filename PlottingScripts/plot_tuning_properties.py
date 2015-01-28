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
import pylab
import numpy as np
import sys
import os
import utils
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from FigureCreator import plot_params
pylab.rcParams.update(plot_params)

class Plotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        tp_fn = self.params['tuning_prop_exc_fn']
        assert (os.path.exists(tp_fn)), 'plot_tuning_properties can only be run on a folder that has created and written the tuning properties to file already.'
        print 'Loading', tp_fn
        self.tp = np.loadtxt(tp_fn)
        print 'Loading', self.params['receptive_fields_exc_fn']
        self.rfs = np.loadtxt(self.params['receptive_fields_exc_fn'])


    def plot_tuning_prop(self):

        tp = self.tp
        fig = pylab.figure(figsize=(12, 12))

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title('Distribution of spatial receptive fields')
        ax1.scatter(tp[:, 0], tp[:, 1], marker='o', c='k')

        ax3.set_title('Histogram of spatial receptive fields')
        cnt, bins = np.histogram(tp[:, 0], bins=20)
        ax3.bar(bins[:-1], cnt, width=bins[1]-bins[0])

        ax2.set_title('Distribution of speed tunings')
        ax2.scatter(tp[:, 2], tp[:, 3], marker='o', c='k')

        ax4.set_title('Histogram of speed tunings')
        cnt, bins = np.histogram(tp[:, 2], bins=20)
        ax4.bar(bins[:-1], cnt, width=bins[1]-bins[0])




    def plot_tuning_space(self):

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Receptive field center $x$', fontsize=18)
        ax.set_ylabel('Preferred speed', fontsize=18)
        patches = []
        for gid in xrange(self.tp[:, 0].size):
            ax.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c='k', markersize=2)
            ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2])
            patches.append(ellipse)

        collection = PatchCollection(patches, alpha=0.1)
        ax.add_collection(collection)
        ylim = ax.get_ylim()
        ax.set_ylim((1.1 * ylim[0], 1.1 * ylim[1]))
#        ax.set_ylim((-3, 3))
        output_fn = self.params['figures_folder'] + 'tuning_space.png'
        print 'Saving to:', output_fn
        pylab.savefig(output_fn, dpi=200)



    def plot_tuning_curves(self, idx):
        """
        idx: integer between 0 - 3 for the column in tuning properties, i.e. the dimension to be plotted
        """
        fig = pylab.figure()
        ax = fig.add_subplot(111)
#        n_cells = self.params['n_exc_mpn']
        n_cells = min(200, self.params['n_exc_mpn'])
#        assert (n_cells  == self.rfs[:, 0].size), 'Mismatch in parameters given to plot_tuning_properties and simulation_parameters.py'
        n_dots = 100 # min(100, n_cells) # for one curve
#        n_rnd = n_cells
#        rnd_gids = np.random.randint(0, self.params['n_exc_mpn'], n_rnd)
        rnd_gids = np.random.choice(xrange(0, self.params['n_exc_mpn']), n_cells)

        color_code = self.tp[rnd_gids, idx]
        norm = matplotlib.colors.Normalize(vmin=color_code.min(), vmax=color_code.max())
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)
        rgba_colors = m.to_rgba(color_code)
        print 'idx', idx
        for i_, gid in enumerate(rnd_gids):
            mu, sigma = self.tp[gid, idx], self.rfs[gid, idx]
            x_ = np.linspace(mu - 3 * sigma, mu + 3 * sigma, n_dots)
            y = np.exp(-.5 * (x_ - mu)**2 / sigma**2)
#            print 'GID %d\tmu %.2e sigma %.2e\tL_max = %.2e, 1/L_max = %.2e' % (i_, mu, sigma, y.max(), 1 / y.max())
            ax.plot(x_, y, color=rgba_colors[i_])

        if idx == 0:
            ax.set_xlabel('RF x-position')
        elif idx == 1:
            ax.set_xlabel('RF y-position')
        elif idx == 2:
            ax.set_xlabel('Speed in x-direction')
        elif idx == 3:
            ax.set_xlabel('Speed in y-direction')
        ax.set_ylabel('Poisson rate envelope')


    def plot_tuning_width_distribution(self):
        """
        Creates a two dimensional plot with rf_center on x-axis and rf-size on the y-axis
        """
        n_cells = self.params['n_exc_mpn']
        assert (n_cells  == self.rfs[:, 0].size), 'Mismatch in parameters given to plot_tuning_properties and simulation_parameters.py'

        fig = pylab.figure()
        pylab.subplots_adjust(hspace=0.5)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        x_axis = self.tp[:, 0]
        y_axis = self.rfs[:, 0]
        ax1.plot(x_axis, y_axis, marker='o', linestyle='None', markersize=2, c='k')
        ax1.set_xlabel('RF_x center')
        ax1.set_ylabel('RF_x size')
        ax1.set_title('Preferred position tuning widths')

        x_axis = self.tp[:, 2]
        y_axis = self.rfs[:, 2]
        ax2.plot(x_axis, y_axis, marker='o', linestyle='None', markersize=2, c='k')
        ax2.set_title('Preferred speed tuning widths')
        ax2.set_xlabel('RF_vx center')
        ax2.set_ylabel('RF_vx size')

        # alternatively use 
        # for all gid: mpatches.Ellipse(self, xy, width, height, angle=0.0, **kwargs)


    def plot_relative_pos_error(self):
#        x = np.around(np.unique(np.abs(.5 - np.sort(self.tp[:, 0]))), decimals=4)
        x = self.tp[:, 0]
        print 'x unique', np.unique(x)
        diff = np.abs(x[1:] - x[:-1])
        rel_diff = diff / x[1:]

        print 'diff x', diff
        print 'rel_diff x', rel_diff
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(rel_diff.size), rel_diff)
        ax.set_title('relative error x')


    def plot_relative_v_error(self):
#        x = np.around(np.unique(np.sort(self.tp[:, 2])), decimals=4)
        x = self.tp[:, 2]
        print 'v unique', np.unique(x)
        diff = x[1:] - x[:-1]
        rel_diff = diff / x[1:]
        print 'diff v ', diff
        print 'rel_diff v ', rel_diff
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(rel_diff.size), rel_diff)
        ax.set_title('relative error v')


    def plot_rf_size_vs_pos(self):

        x = self.tp[:, 0]
        y = self.rfs[:, 0]
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'o', markersize=2)
        ax.set_title('Receptive field sizes for position')
        ax.set_xlabel('RF_x position')
        ax.set_ylabel('RF_x size')


    def plot_rf_size_vs_speed(self):

        x = self.tp[:, 2]
        y = self.rfs[:, 2]
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'o', markersize=2)
        ax.set_title('Receptive field sizes for speed')
        ax.set_xlabel('RF_v position')
        ax.set_ylabel('RF_v size')


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

    
    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_tuning_prop()
    Plotter.plot_tuning_space()
#    Plotter.plot_tuning_curves(0)
#    Plotter.plot_tuning_curves(2)
    Plotter.plot_tuning_width_distribution()

    Plotter.plot_relative_pos_error()
    Plotter.plot_relative_v_error()

#    Plotter.plot_rf_size_vs_pos()
#    Plotter.plot_rf_size_vs_speed()

    pylab.show()
