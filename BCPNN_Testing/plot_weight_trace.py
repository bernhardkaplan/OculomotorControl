import sys
import numpy as np
import pylab

fn = sys.argv[1]

d = np.loadtxt(fn)

fig = pylab.figure()

ax = fig.add_subplot(111)

x_axis = np.arange(d[:, 0].size)
ax.plot(x_axis, d[:, 1])

pylab.show()
