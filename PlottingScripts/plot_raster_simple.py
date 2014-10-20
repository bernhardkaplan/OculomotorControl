import pylab
import numpy
import sys
import os


#if (len(sys.argv) < 2):
#    fn = raw_input("Please enter data file to be plotted\n")
#else:
#    fns = sys.argv[1]

fig = pylab.figure()
ax = fig.add_subplot(111)

for fn in sys.argv[1:]:

    file_size = os.path.getsize(fn)
    if file_size > 0:
        data = pylab.loadtxt(fn)

        print 'debug', fn
        if (data.ndim == 1):
    #        x_axis = numpy.arange(data.size)
        #    pylab.plot(x_axis, data)
    #        pylab.scatter(x_axis, data)
    #        pylab.plot(data[1], data[0], 'o', markersize=1, color='k')
            ax.plot(data[1], data[0], 'o', markersize=1, color='k')

        else:
        #    pylab.scatter(data[:,0], data[:,1])
            ax.plot(data[:,1], data[:,0], 'o', markersize=1, color='k')

        ax.set_title(fn)



    #pylab.xlim((0, 1200))

pylab.show()
