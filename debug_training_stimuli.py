import pylab
import numpy as np
import matplotlib
import sys

#fn = 'training_stimuli_nV16_nX20_seed2.dat'
#fn = 'Training_DEBUG_titer25_nStim320_0-320_gain0.80_seeds_111_2/Data/training_stimuli_parameters.txt'
#fn = 'training_stimuli_1200.txt'

fn = sys.argv[1]
d = np.loadtxt(fn)

fig = pylab.figure()
ax = fig.add_subplot(111)


N = d[:, 0].size
N = 300
bounds = range(N)

cmap = matplotlib.cm.jet
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
m.set_array(np.arange(bounds[0], bounds[-1], 1.))
rgba_colors = m.to_rgba(bounds)

for i_ in xrange(N):
    ax.plot(d[i_, 0], d[i_, 2], 'o', color=rgba_colors[i_])
    ax.text(d[i_, 0], d[i_, 2], '%d' % i_, color=rgba_colors[i_])

ax.set_xlim((0, 1))
ax.set_ylim((-2, 2))
pylab.show()
