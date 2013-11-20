import sys
import pylab
import numpy as np
import utils




if __name__ == '__main__':

    fns = sys.argv[1:]
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for fn in fns:
        d = np.loadtxt(fn)
        gid = 'all'
        if d.size > 0:
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
                time_axis, volt = utils.extract_trace(d, gid)
                ax.plot(time_axis, volt, label='%d' % gid, lw=2)

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    pylab.legend()
    pylab.show()
