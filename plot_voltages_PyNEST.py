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

    pylab.legend()
    pylab.show()
