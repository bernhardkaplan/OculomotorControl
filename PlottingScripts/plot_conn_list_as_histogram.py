import numpy
import pylab
import sys

fns = sys.argv[1:]

for fn in fns:
    title = fn

    d = numpy.loadtxt(fn, skiprows=1)
    #d = numpy.loadtxt(fn)

    #print "min weight:", d[:,2].min()
    #print "max weight:", d[:,2].max()
    #print "mean weight:", d[:,2].mean()

    #print "exp min weight:", numpy.exp(d[:,2].min())
    #print "exp max weight:", numpy.exp(d[:,2].max())
    #print "exp mean weight:", numpy.exp(d[:,2].mean())
    fig = pylab.figure()

    bin_width = 5e-2
    x_max = d[:,2].max()
    data, bins = numpy.histogram(d[:,2], x_max/bin_width)
    ax = fig.add_subplot(111)
    ax.bar(bins[:-1], data, width=bin_width/2.)
    pylab.title("Bin width = %f" % (bin_width))
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

