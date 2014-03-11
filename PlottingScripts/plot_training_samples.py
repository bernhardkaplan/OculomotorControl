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
pylab.rcParams.update(plot_params)

class Plotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        tp_fn = self.params['tuning_prop_exc_fn']
        print 'Loading', tp_fn
        self.tp = np.loadtxt(tp_fn)
        print 'Loading', self.params['receptive_fields_exc_fn']
        self.rfs = np.loadtxt(self.params['receptive_fields_exc_fn'])


    def plot_training_sample_space(self, plot_process=False):

        if plot_process:
            d = np.loadtxt(self.params['motion_params_fn'])
        else:
            d = np.loadtxt(self.params['training_sequence_fn'])

        # get the start value of a training stimulus 
#        zero_idx = (d[:, 2] == 0).nonzero()[0]

        fig = pylab.figure()#figsize=(12, 12))
        ax1 = fig.add_subplot(111)

        # plot the stimulus start points
        for i_ in xrange(self.params['n_stim']):
                
            if plot_process:
                idx = i_ * self.params['n_iterations_per_stim']
                mp = d[idx, :]
                ax1.plot(mp[0], mp[2], 'o', markersize=5, color='r')
                idx_stop = (i_ + 1) * self.params['n_iterations_per_stim']
                mps = d[idx:idx_stop, :]
                ax1.plot(mps[:, 0], mps[:, 2], '--', color='r', lw=3)
            else:
                mp = d[i_, :]
                ax1.plot(mp[0], mp[2], 'o', markersize=5, color='r')

#            for j_ in xrange(self.params['n_iterations_per_stim']):

        patches = []
#        fig2 = pylab.figure()
        for gid in xrange(self.params['n_exc_mpn']):
            ax1.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c='k', markersize=2)
            ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2])
            patches.append(ellipse)

        collection = PatchCollection(patches, alpha=0.1)
#        collection.set_array(
        ax1.add_collection(collection)

        ax1.set_title('Training stimuli state space')
        ax1.set_xlabel('Stimulus position') 
        ax1.set_ylabel('Stimulus speed vx') 
        output_fig = params['figures_folder'] + 'stimulus_state_space.png'
        print 'Saving to:', output_fig
        pylab.savefig(output_fig, dpi=200)




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
    Plotter.plot_training_sample_space()
    pylab.show()
