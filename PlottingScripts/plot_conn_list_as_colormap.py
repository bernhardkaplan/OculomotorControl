import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np 
import utils
import json
#from matplotlib import cm

def load_params_from_folder(folder_name):
    param_fn = folder_name + '/Parameters/simulation_parameters.json'
    f = file(param_fn, 'r')
    print 'Loading parameters from', param_fn
    params = json.load(f)
    return params


def plot_matrix(d, title=None, clim=None):

    cmap_name = 'bwr'
    if clim != None:
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])#, clip=True)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
        m.set_array(np.arange(clim[0], clim[1], 0.01))
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    print "plotting .... "
    if clim != None:
        cax = ax.pcolormesh(d, cmap=cmap_name, vmin=clim[0], vmax=clim[1])
    else:
        cax = ax.pcolormesh(d, cmap=cmap_name)
    ax.set_ylim((0, d.shape[0]))
    ax.set_xlim((0, d.shape[1]))
    if title != None:
        ax.set_title(title)
    pylab.colorbar(cax)
    return ax


def check_is_it_d1(fn):
    if fn.find('d1') != -1:
        return 'd1'
    elif fn.find('d2') != -1:
        return 'd2'



def plot_conn_list(conn_list_fn, params=None, clim=None, src_cell_type=None):
    if not os.path.exists(conn_list_fn):
        print 'Merging default connection files...'
        merge_pattern = params['mpn_bgd1_conn_fn_base']
        fn_out = params['mpn_bgd1_merged_conn_fn']
        utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)
        if params['with_d2']:
            merge_pattern = params['mpn_bgd2_conn_fn_base']
            fn_out = params['mpn_bgd2_merged_conn_fn']
            utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)

    print 'Loading ', conn_list_fn
    data = np.loadtxt(conn_list_fn)

    src_min, src_max= np.min(data[:, 0]), np.max(data[:, 0])
    n_src = src_max - src_min + 1
    tgt_min, tgt_max= np.min(data[:, 1]), np.max(data[:, 1])
    n_tgt = tgt_max - tgt_min + 1

    conn_mat = np.zeros((n_src, n_tgt))

    print 'src_min, src_max', src_min, src_max
    print 'tgt_min, tgt_max', tgt_min, tgt_max
    print 'debug conn_mat shape', conn_mat.shape
    for c in xrange(data[:,0].size):
        src = data[c, 0] - src_min
        tgt = data[c, 1] - tgt_min
#        print 'src, tgt', src, tgt, data[c, 1]
        conn_mat[src, tgt] = data[c, 2]

    ax = plot_matrix(conn_mat, clim=clim)
    print 'connmat min max', np.min(conn_mat), np.max(conn_mat)
#    ax.set_xlabel('Target: %s' % tgt_cell_type)
#    ax.set_ylabel('Sources: %s' % src_cell_type)
    if params != None:
        output_fn = params['figures_folder'] + conn_list_fn.rsplit("/")[-1].rsplit(".txt")[0] + "_cmap.png"
        title = output_fn.rsplit('/')[-1]
        ax.set_title(title)
        print "Saving fig to:", output_fn
        pylab.savefig(output_fn)
    return conn_mat


if __name__ == '__main__':

    """usage:
    python PlottingScripts/plot_conn_list_as_colormap.py Training_taup225_nStim1_nExcMpn800_nStates14_nActions15_it15-225_wMPN-BG3.50/Connections/merged_mpn_bg_d2_connection_dev_it*.txt
    """

    fns = sys.argv[1:] # 
    if len(sys.argv) > 2:
        for fn in fns:
            params = utils.load_params(fn)
            tgt_type = 'd1'
            conn_list_fn = params['mpn_bg%s_merged_conn_fn' % tgt_type]
            plot_conn_list(conn_list_fn, params=params, clim=clim)
            if params['with_d2']:
                tgt_type = 'd2'
                conn_list_fn = params['mpn_bg%s_merged_conn_fn' % tgt_type]
                plot_conn_list(conn_list_fn, params=params, clim=clim)
    elif len(sys.argv) == 2:
        if sys.argv[1].endswith('.json') or os.path.isdir(sys.argv[1]):
            params = utils.load_params(sys.argv[1])
            tgt_type = 'd1'
            print 'Plotting connections targeting :', tgt_type
            conn_list_fn = params['mpn_bg%s_merged_conn_fn' % tgt_type]
#            clim = (-5., 5.)
            clim = None
            plot_conn_list(conn_list_fn, params=params, clim=clim)
            if params['with_d2']:
                tgt_type = 'd2'
                print 'Plotting connections targeting :', tgt_type
                conn_list_fn = params['mpn_bg%s_merged_conn_fn' % tgt_type]
                clim = (-3., 3.)
                plot_conn_list(conn_list_fn, params=params, clim=clim)
    #         params = load_params_from_folder(fn)
    #         plot_conn_list(params=params)
        else:          
            conn_list_fn = sys.argv[1]
            plot_conn_list(conn_list_fn, params=None)#, clim=clim)

    
    pylab.show()
