import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import pylab
import numpy as np
import utils


if __name__ == '__main__':

    print 'Usage: \nplot_voltages_simple.py [FOLDER_NAME] [GID_1] [GID_2]'
    params = utils.load_params(sys.argv[1])

    if len(sys.argv) > 2:
        bg_or_mpn = sys.argv[2]
    else:
#        bg_or_mpn = 'mpn'
        bg_or_mpn = 'bg'
    if bg_or_mpn == 'bg':
        volt_fn_base = params['bg_volt_fn']
    else:
        volt_fn_base = params['mpn_exc_volt_fn']
    fns = utils.find_files(params['spiketimes_folder'], volt_fn_base)
    fig = pylab.figure()
    ax = fig.add_subplot(111)

    gids = params['gids_to_record_%s' % bg_or_mpn]

    print 'GIDS:', gids
#    gids = [4966]
    v_mean = {}

    files = {}
    for gid in gids:
        gid = int(gid)
        print 'GID ', gid
        for fn in fns:
            if fn not in files.keys():
                print 'Loading', fn
                d = np.loadtxt(params['spiketimes_folder'] + fn)
                files[fn] = d
            else:
                d = files[fn]
            time_axis, volt = utils.extract_trace(d, gid)
            if volt.size > 0:
                ax.plot(time_axis, volt, label='%d' % gid, lw=2)
                v_mean[gid] = volt.mean()
    for gid in gids:
        print 'GID v_mean:', gid, v_mean[gid]


    if params['training']:
        title = 'Training'
    else:
        title = 'Testing'
    ax.set_title(title)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    pylab.legend()
    output_fn = params['figures_folder'] + 'bg_voltages.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn)
    pylab.show()

