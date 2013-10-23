import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import matplotlib
import pylab
import numpy as np
import sys
import os
import utils


class Plotter(object):

    def __init__(self, params, it_max=None):
        self.params = params


    def plot_tuning_prop(self):
        tp_fn = self.params['tuning_prop_exc_fn']
        print 'Loading', tp_fn
        tp = np.loadtxt(tp_fn)

        fig = pylab.figure()

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
    pylab.show()
