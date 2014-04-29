import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import pylab
import utils

fns = sys.argv[1:]



for fn in fns:
    title = fn

    d = np.loadtxt(fn, skiprows=1)

    weights = d[:, 2]

    thresholds = np.arange(0.00, 0.3, 0.01)
    n_total = weights.size
    n_realized = np.zeros(thresholds.size)
    for i_, thresh in enumerate(thresholds):
        n_realized[i_] = utils.threshold_array(weights, thresh, absolute_val=False)[0].size

    n_realized /= float(n_total)
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, n_realized)
    ax.set_xlabel('Weight Threshold')
    ax.set_ylabel('Percentage of realized connections')

    #d = np.loadtxt(fn)

    #print "min weight:", d[:,2].min()
    #print "max weight:", d[:,2].max()
    #print "mean weight:", d[:,2].mean()

    #print "exp min weight:", np.exp(d[:,2].min())
    #print "exp max weight:", np.exp(d[:,2].max())
    #print "exp mean weight:", np.exp(d[:,2].mean())
    fig = pylab.figure()

    bin_width = .1
    x_max = d[:,2].max()
    data, bins = np.histogram(d[:,2], x_max/bin_width)
    ax = fig.add_subplot(111)
    ax.bar(bins[:-1], data, width=bin_width/2.)
    pylab.title("%s\nBin width = %f" % (title, bin_width))
    pylab.xlabel("Weight")
    pylab.ylabel("Count")
    #pylab.ylim(
    if fn.find('.dat') != -1:
        output_fn = fn.rsplit('.dat')[0] + '.png'
    elif fn.find('.txt') != -1:
        output_fn = fn.rsplit('.txt')[0] + '.png'
    print output_fn


    pylab.savefig(output_fn)

pylab.show()

