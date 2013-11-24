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
import json
from FigureCreator import plot_params
pylab.rcParams.update(plot_params)
from PlotMPNActivity import ActivityPlotter

class Plotter(object):

    def __init__(self, training_params, test_params):
        self.training_params = training_params
        self.test_params = test_params

    def compare_xdisplacement(self):
        training_fn = self.training_params['data_folder'] + 'mpn_xdisplacement.dat'
        print 'Loading data:', training_fn
        training_d= np.loadtxt(training_fn)
        training_xdispl = np.abs(training_d[:, 0] - .5)

        test_fn = self.test_params['data_folder'] + 'mpn_xdisplacement.dat'
        print 'Loading data:', test_fn
        test_d= np.loadtxt(test_fn)
        test_xdispl = np.abs(test_d[:, 0] - .5)

        t_axis = test_d[:, 4]
        t_axis += .5 * self.test_params['t_iteration']
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Comparison of training vs. test performance')

        p1, = ax.plot(t_axis, training_xdispl, c='k', lw=3)
        p2, = ax.plot(t_axis, test_xdispl, c='b', ls='--', lw=3)
        plots = [p1, p2]

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Retinal displacement')
        ax.legend(plots, ['training', 'test'], loc='upper left')

        self.AP = ActivityPlotter(self.test_params)
        self.AP.plot_vertical_lines(ax)

        output_fn = self.test_params['figures_folder'] + 'comparison_training_test_xdisplacement.png'
        print 'Saving figure to:', output_fn
        pylab.savefig(output_fn, dpi=300)



if __name__ == '__main__':


    training_folder = sys.argv[1]
    test_folder = sys.argv[2]
    print 'Loading data from:'


    training_param_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    f = file(training_param_fn, 'r')
    print 'Loading training parameters from', training_param_fn
    training_params = json.load(f)
    
    test_param_fn = os.path.abspath(test_folder) + '/Parameters/simulation_parameters.json'
    f = file(test_param_fn, 'r')
    print 'Loading test parameters from', test_param_fn
    test_params = json.load(f)

    Plotter = Plotter(training_params, test_params)
    Plotter.compare_xdisplacement()

    pylab.show()
