import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import simulation_parameters
import utils
import numpy as np
import json
import pylab
import FigureCreator


def plot_action_indices_histogram(params):
    fn = params['action_indices_fn']
    d = np.loadtxt(fn)
    cnt, bins = np.histogram(d, bins=params['n_actions'], range=(0, params['n_actions']))
    print 'Action hist bins', bins
    print 'Action hist cnt', cnt
    print 'to be completed...',


def plot_histogram(fn):

    d = np.loadtxt(fn)
    n_bins = (np.max(d) - np.min(d))
    cnt, bins = np.histogram(d, bins=n_bins, range=(np.min(d), np.max(d)))
    print 'First half:', bins[:n_bins/2]
    print 'Sum first half:', cnt[:n_bins/2].sum()
    print 'Second half:', bins[-(n_bins/2 + 1):]
    print 'Sum second half:', cnt[-(n_bins/2 + 1):].sum()

#    data = np.loadtxt(fn)
#    d = data[:, 2]
#    n_bins = 100
#    cnt, bins = np.histogram(d, bins=n_bins, range=(np.min(d), np.max(d)))
#    print 'First half:', bins[:49] 
#    print '1 sum:', cnt[:49].sum()
#    print 'Second half:', bins[-52:]
#    print '2 sum:', cnt[-52:].sum()

    print 'hist bins', bins
    print 'hist cnt', cnt
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(bins[:-1], cnt, width=bins[1]-bins[0])
    ax1.set_xlabel('Actions taken')
    ax1.set_ylabel('Count')
#    ax1.set_xlim((0, params['n_actions']))


if __name__ == '__main__':
    try:
        folder = os.path.abspath(sys.argv[1])
        params = utils.load_params(folder)
    except:
        fn = sys.argv[1]
        plot_histogram(fn)


    pylab.show()
#print 'Action mapping', action_mapping
#idx_never_done = np.nonzero(cnt == 0)[0]
#print 'Actions never done:', idx_never_done
#print 'Actions never done (vx):', action_mapping[idx_never_done]
