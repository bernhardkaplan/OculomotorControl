import sys
import pylab
import numpy as np
import utils




if __name__ == '__main__':

    print 'Usage: \nplot_voltages_simple.py [FOLDER_NAME] [GID_1] [GID_2]'
    params = utils.load_params(sys.argv[1])

    volt_fn_base = params['bg_volt_fn']
    fns = utils.find_files(params['spiketimes_folder'], '(.*volt.*)')
#    fns = utils.find_files(params['spiketimes_folder'], 'mpn_volt')
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for gid in sys.argv[2:]:
        gid = int(gid)
        print 'DEBUG', gid
        for fn in fns:
            print 'debug', fn
            d = np.loadtxt(params['spiketimes_folder'] + fn)
            time_axis, volt = utils.extract_trace(d, gid)
            if volt.size > 0:
                ax.plot(time_axis, volt, label='%d' % gid, lw=2)

#    for fn in fns:
#        d = np.loadtxt(fn)
#        gid = 'all'
#        if d.size > 0:
#            if gid == None:
#                recorded_gids = np.unique(d[:, 0])
#                gids = random.sample(recorded_gids, n)
#                print 'plotting random gids:', gids
#            elif gid == 'all':
#                gids = np.unique(d[:, 0])
#            elif type(gid) == type([]):
#                gids = gid
#            else:
#                gids = [gid]
#                
#            for gid in gids:
#                time_axis, volt = utils.extract_trace(d, gid)
#                ax.plot(time_axis, volt, label='%d' % gid, lw=2)

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    pylab.legend()
    pylab.show()
