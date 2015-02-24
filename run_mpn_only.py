import sys
import os
import numpy as np
import json
import nest
import VisualInput
import MotionPrediction
import BasalGanglia
import simulation_parameters
import CreateConnections
import utils
from PlottingScripts.PlotMPNActivity import MetaAnalysisClass
from PlottingScripts.PlotMPNActivity import ActivityPlotter
from copy import deepcopy
import time
import pylab
import matplotlib.mlab as mlab

try: 
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"




def plot_spike_histogram(params):

    merged_spike_fn = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    if not os.path.exists(merged_spike_fn):
        utils.merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], merged_spike_fn)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    sort_idx = 0
    tp_idx_sorted = tp[:, sort_idx].argsort() # + 1 because nest indexing
    merged_spike_fn = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']

    print 'Loading:', merged_spike_fn
    spikes_unsrtd = np.loadtxt(merged_spike_fn)

    nspikes = utils.get_spikes(merged_spike_fn, n_cells=params['n_exc_mpn'], get_spiketrains=False, gid_idx=0, NEST=True)

    nspikes_sorted = np.zeros(params['n_exc_mpn'])

    for i_, gid in enumerate(tp_idx_sorted):
        nspikes_sorted[i_] = nspikes[gid]

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(params['n_exc_mpn']), nspikes_sorted, width = 1)

    new_xticklabels = []
#    old_xticks = ax.get_xticks()
#    xticks = np.linspace(0, params['n_exc_mpn']-1, 5, endpoint=True)
#    for xt_ in old_xticks[:-1]:
#        new_xticklabels.append('%.1f' % (tp[tp_idx_sorted[int(xt_)], sort_idx]))
#        print 'debug', tp[tp_idx_sorted[int(xt_)], sort_idx]

    xticks = []
    for i_ in np.linspace(0, params['n_exc_mpn'] - 1, 10, endpoint=True):
#        print 'debug i_', i_, tp[tp_idx_sorted[int(i_)], sort_idx]
        new_xticklabels.append('%.1f' % (tp[tp_idx_sorted[int(i_)], sort_idx]))
        xticks.append(tp_idx_sorted[int(i_)])
    ax.set_xticks(xticks)
    ax.set_xticklabels(new_xticklabels)


    # gaussian in same plot with different x-axis for simplicity
    mu = tp[tp_idx_sorted[np.argmax(nspikes_sorted)], sort_idx]
#    print 'debug', np.argmax(nspikes_sorted)
#    print 'debug2 ', tp_idx_sorted
#    print 'mu:', mu
#    ax.plot(np.arange(0., 1., 100), mlab.normpdf(np.arange(0., 1., 100), mu, params['blur_X'] + 1e-6), ls='--', c='r')

    ax.set_xlim((0, params['n_exc_mpn']))
    ax.set_ylabel('Number of spikes')
    ax.set_xlabel('Tuning properties')
    output_fn = params['figures_folder'] + 'nspike_histogram_sort_idx%d_bX%.1f_bV%.1f.png' % (sort_idx, params['blur_X'], params['blur_V'])
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)



if __name__ == '__main__':

    plotting = True
    run = True
    stim_params = [0.5, 0.5, 1.0, 0.]
    blur = [0.05, 0.05]
    t_sim = 0
    GP = simulation_parameters.global_parameters()
    params = GP.params
    if run:
        params['initial_state'] = stim_params
        params['blur_X'] = blur[0]
        params['blur_V'] = blur[1]
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation

        if pc_id == 0:
            utils.remove_files_from_folder(params['spiketimes_folder'])
            utils.remove_files_from_folder(params['input_folder_mpn'])
        if comm != None:
            comm.Barrier()
        VI = VisualInput.VisualInput(params, comm=comm)
        MT = MotionPrediction.MotionPrediction(params, VI, comm)
        VI.current_motion_params = deepcopy(stim_params)
        stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
        MT.update_input(stim)
        if comm != None:
            comm.Barrier()
        nest.Simulate(params['t_iteration'])
        t_sim += params['t_iteration']
        MT.advance_iteration(params['t_iteration'])
        VI.advance_iteration(params['t_iteration'])
        if comm != None:
            comm.Barrier()
        v_eye = [0., 0.]
        stim, supervisor_state = VI.compute_input(MT.local_idx_exc, v_eye, params['t_iteration'])
        print 'Saving spike trains...'
        iteration_cnt = 0
        utils.save_spike_trains(params, iteration_cnt, stim, MT.local_idx_exc)
        MT.update_input(stim)
        if comm != None:
            comm.Barrier()
        nest.Simulate(params['t_iteration'])
        t_sim += params['t_iteration']
        if comm != None:
            comm.Barrier()

    if pc_id == 0 and plotting:

        plot_params = {'backend': 'png',
                      'axes.labelsize': 20,
                      'axes.titlesize': 20,
                      'text.fontsize': 20,
                      'xtick.labelsize': 16,
                      'ytick.labelsize': 16,
                      'legend.pad': 0.2,     # empty space around the legend box
                      'legend.fontsize': 14,
                       'lines.markersize': 1,
                       'lines.markeredgewidth': 0.,
                       'lines.linewidth': 1,
                      'font.size': 12,
                      'path.simplify': False,
                      'figure.subplot.left':.15,
                      'figure.subplot.bottom':.15,
                      'figure.subplot.right':.94,
                      'figure.subplot.top':.84,
                      'figure.subplot.hspace':.30,
                      'figure.subplot.wspace':.30}

        pylab.rcParams.update(plot_params)

        plot_spike_histogram(params) 
        iter_range = (0, 3)
        it_max = 1
        AP = ActivityPlotter(params, it_max)
        AP.plot_raster_sorted(title='', cell_type='exc', sort_idx=0, t_range=[0, t_sim])

        AP.bin_spiketimes()
        AP.plot_input(v_or_x='x')
        AP.plot_input(v_or_x='v')
        AP.plot_input_cmap(iteration=0, t_plot=params['t_iteration'])
        AP.plot_output(iter_range, v_or_x='x', compute_state_differences=False)
        AP.plot_output_xv_cmap(t_plot=params['t_iteration'])
#        MAC = MetaAnalysisClass([params['folder_name']], show=True)

    else:
        print 'Waiting for plotting...'

    pylab.show()
