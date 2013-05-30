import pylab
import numpy as np
import sys

if (len(sys.argv) < 2):
    fn = raw_input("Please enter data file to be plotted\n")
else:
    fn = sys.argv[1]

data = np.load(fn)
#data = np.loadtxt(fn)
if (data.ndim == 1):
    x_axis = np.arange(data.size)
    pylab.plot(x_axis, data)
else:
    for i in xrange(data[:, 0].size):
        pylab.plot(data[i, 0], data[i, 1], 'o')
#    pylab.plot(data[:,0], data[:, 1], '-')
#    pylab.plot(data[:,0], data[:, 1] / data[:, 0], '-')
#    pylab.plot(data[:,0], data[:, 2], '-')
#    pylab.plot(data[:,0], data[:, 3], '-')

#pylab.plot((0.12, 0.12), (0, 7000), lw=2, c='k')
pylab.show()
