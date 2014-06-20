import pylab
import numpy
import sys


#if (len(sys.argv) < 2):
#    fn = raw_input("Please enter data file to be plotted\n")
#else:
#    fns = sys.argv[1]

for fn in sys.argv[1:]:
    data = pylab.loadtxt(fn)

    if (data.ndim == 1):
        x_axis = numpy.arange(data.size)
    #    pylab.plot(x_axis, data)
        pylab.scatter(x_axis, data)
    else:
    #    pylab.scatter(data[:,0], data[:,1])
        pylab.plot(data[:,1], data[:,0], 'o', markersize=1, color='k')



    #pylab.xlim((0, 1200))

pylab.show()
