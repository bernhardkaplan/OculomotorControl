import sys
import pylab
import numpy as np

def extract_trace(d, gid):
    """
    d : voltage trace from a saved with compatible_output=False
    gid : cell_gid
    """
    mask = gid * np.ones(d[:, 0].size)
    indices = mask == d[:, 0]
    time_axis, volt = d[indices, 1], d[indices, 2]
    return time_axis, volt

def plot_average_volt(fn, gid=None, n=1):
    print 'Plotting average voltage; loading', fn
    d = np.loadtxt(fn)
    if gid == None:
        gid_range = np.unique(d[:, 0])
        gids = np.random.randint(np.min(gid_range), np.max(gid_range) + 1, n)
        print 'plotting random gids:', gids
    elif gid == 'all':
        gids = np.unique(d[:, 0])
    elif type(gid) == type([]):
        gids = gid
    else:
        gids = [gid]
    
    time_axis, volt = extract_trace(d, gids[0])
    all_volt = np.zeros((time_axis.size, len(gids)))

    for i_, gid in enumerate(gids):
        time_axis, volt = extract_trace(d, gid)
        print 'gid %d v_mean, std = %.2f +- %.2f; min %.2f max %.2f, diff %.2f ' % (gid, volt.mean(), volt.std(), volt.min(), volt.max(), volt.max() - volt.min())
        all_volt[:, i_] = volt

    avg_volt = np.zeros((time_axis.size, 2))
    for t in xrange(time_axis.size):
        avg_volt[t, 0] = all_volt[t, :].mean()
        avg_volt[t, 1] = all_volt[t, :].std()

    print 'Average voltage and std: %.2e +- %.2e (%.2e)' % (avg_volt[:, 0].mean(), avg_volt[:, 0].std(), avg_volt[:, 1].mean())
    pylab.errorbar(time_axis, avg_volt[:, 0], yerr=avg_volt[:, 1], lw=3, c='k') 





if __name__ == '__main__':

    fn = sys.argv[1]
    d = np.loadtxt(fn)
    gid = 'all'
    if gid == None:
        recorded_gids = np.unique(d[:, 0])
        gids = random.sample(recorded_gids, n)
        print 'plotting random gids:', gids
    elif gid == 'all':
        gids = np.unique(d[:, 0])
    elif type(gid) == type([]):
        gids = gid
    else:
        gids = [gid]
        
    for gid in gids:
        time_axis, volt = extract_trace(d, gid)
        pylab.plot(time_axis, volt, label='%d' % gid, lw=2)

    pylab.show()
