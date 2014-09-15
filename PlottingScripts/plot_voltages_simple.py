import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import pylab
import numpy as np
import utils
colorlist = utils.get_colorlist()
n_colors = len(colorlist)

#def get_gids_from_filenames(fns):
#    for fns 

if __name__ == '__main__':


    new_fig = False
    if new_fig == False:
        fig = pylab.figure()
        ax = fig.add_subplot(111)

    for fn in sys.argv[1:]:
#    fn = sys.argv[1]
        file_size = os.path.getsize(fn)
        if file_size != 0:
            
            d = np.loadtxt(fn)
            gids = np.unique(d[:, 0])

            print 'GIDS:', gids
            if new_fig:
                fig = pylab.figure()
                ax = fig.add_subplot(111)
            for gid in gids:
                time_axis, volt = utils.extract_trace(d, gid)
                ax.plot(time_axis, volt, label='%d' % gid, lw=2)

            title = fn.rsplit('/')[-1]
            ax.set_title(title)
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Volt [mV]')

            if new_fig:
                pylab.legend()
    if not new_fig:
        pylab.legend()
    pylab.show()

#    print 'Usage: \nplot_voltages_simple.py [FOLDER_NAME] [GID_1] [GID_2]'
#    params = utils.load_params(sys.argv[1])

#    if len(sys.argv) > 2:
#        bg_or_mpn = sys.argv[2]
#    else:
#        bg_or_mpn = 'mpn'
#        bg_or_mpn = 'bg'
#    if bg_or_mpn == 'bg':
#        volt_fn_base = params['bg_volt_fn']
#        gids = range(params['n_exc_mpn'] + 1, bg_gid_max)
#    else:
#        volt_fn_base = params['mpn_exc_volt_fn']
#        gids = range(1, params['n_exc_mpn'] + 1)

#    fns = utils.find_files(params['spiketimes_folder'], volt_fn_base)
#    fig = pylab.figure()
#    ax = fig.add_subplot(111)

#    gids = params['gids_to_record_%s' % bg_or_mpn]

#    gids = [4966]
#    v_mean = {}

#    cnt_ = 0
#    files = {}
#    for gid in gids:
#        gid = int(gid)
#        print 'GID ', gid
#        for fn in fns:
#            if fn not in files.keys():
#                print 'Loading', fn
#                d = np.loadtxt(params['spiketimes_folder'] + fn)
#                files[fn] = d
#            else:
#                d = files[fn]
#            time_axis, volt = utils.extract_trace(d, gid)
#            if volt.size > 0:
#                ax.plot(time_axis, volt, label='%d' % gid, lw=2, c=colorlist[cnt_ % n_colors])
#                v_mean[gid] = volt.mean()
#                cnt_ += 1

#    for gid in v_mean.keys():
#        print 'GID v_mean:', gid, v_mean[gid]


#    if params['training']:
#        title = 'Training'
#    else:
#        title = 'Testing'
#    ax.set_title(title)
#    ax.set_xlabel('Time [ms]')
#    ax.set_ylabel('Volt [mV]')
#    pylab.legend()
#    output_fn = params['figures_folder'] + 'bg_voltages.png'
#    print 'Saving figure to:', output_fn
#    pylab.savefig(output_fn)
#    pylab.show()

