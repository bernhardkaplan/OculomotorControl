import pylab
import numpy as np
import utils

x_min = 0.05
x_max = 0.45
n_x = 20
logscale = 2.0


def get_logspace(x_min, x_max, divisor, n, logscale):
     return np.logspace(np.log(x_min) / divisor, np.log(x_max) / divisor, n, base=logscale)

    
def plot_hist(y):
    cnt, bins = np.histogram(y, bins=10)
    pylab.bar(bins[:-1], cnt, width=bins[1]-bins[0])

    
def get_relative_diff(x):
    diff = np.zeros(len(x) - 1)
    diff = (x[:-1] - x[1:]) / x[:-1]
    return diff


y_debug = utils.get_xpos_log_distr(2., n_x, x_min=1e-6, x_max=.5)
y_debug2 = get_logspace(1e-6, .5, np.log(2.), n_x, logscale)

y2 = get_logspace(x_min, x_max, np.log(2.), n_x, 2.)
y2_1 = get_logspace(x_min, x_max, 1, n_x, 2.)
y3 = get_logspace(x_min, x_max, 1., n_x, 3.)
y4 = get_logspace(x_min, x_max, 10., n_x, 1.6)
#y3 = get_logspace(x_min, x_max, 1, n_x, 2.)


pylab.figure()
pylab.title('logspaces')
pylab.plot(range(y2.size), y2, 'o', label='y2')
pylab.plot(range(y2_1.size), y2_1, 'D', label='y2_1')
pylab.plot(range(y3.size), y3, '^', label='y3')
pylab.plot(range(y4.size), y4, 's', label='y4')
pylab.plot(range(y_debug.size), y_debug, '*', label='y_debug')
pylab.plot(range(y_debug2.size), y_debug2, '*', label='y_debug2')
pylab.legend()


#pylab.figure()
#pylab.title('histograms')
#plot_hist(y2)
#plot_hist(y3)


d2  = get_relative_diff(y2)
d2_1  = get_relative_diff(y2_1)
d3  = get_relative_diff(y3)
d4  = get_relative_diff(y4)
d_debug  = get_relative_diff(y_debug)
d_debug2  = get_relative_diff(y_debug2)
print 'debug y_debug', y_debug
print 'debug d_debug', d_debug
print 'debug y_debug2', y_debug2
print 'debug d_debug2', d_debug2

pylab.figure()
pylab.title('relative errors')
pylab.plot(range(len(d2)), d2, 'o', label='y2')
pylab.plot(range(len(d2_1)), d2_1, 'o', label='y2_2')
pylab.plot(range(len(d3)), d3, 'o', label='y3')
pylab.plot(range(len(d4)), d4, 'o', label='y4')
pylab.plot(range(len(d_debug)), d_debug, 'o', label='y_debug')
pylab.plot(range(len(d_debug2)), d_debug2, 'o', label='y_debug2')


pylab.show()
