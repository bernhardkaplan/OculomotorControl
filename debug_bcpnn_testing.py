import os
import sys
import json
import numpy as np
import utils
from PlottingScripts.plot_bcpnn_traces import TracePlotter
from PlottingScripts.plot_voltages import VoltPlotter
import pylab
from PlottingScripts import FigureCreator

FigureCreator.plot_params['figure.subplot.left'] = .17
pylab.rcParams.update(FigureCreator.plot_params)

class DebugTraces(object):

    def __init__(self):
        self.trace_buffer = {}
        self.spike_data = {}
        self.VP = {}

    def check_spike_files(self, params, cell_type_post='d1'): 
        fn_pre = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
        fn_post = params['spiketimes_folder'] + params['%s_spikes_fn_merged_all' % cell_type_post]
        if (not os.path.exists(fn_pre)) or (not os.path.exists(fn_post)):
            utils.merge_spikes(params)
        print 'Debug, spike data exists:', os.path.exists(fn_pre), os.path.exists(fn_post)
        print 'Debug, spike data sizes:', os.path.getsize(fn_pre), os.path.getsize(fn_post)


    def get_weights(self, pre_gids, post_gids, tgt_cell_type='d1'):
        conn_list_fn = training_params['mpn_bg%s_merged_conn_fn' % tgt_cell_type]
        print 'Loading weights from:', conn_list_fn
        if not os.path.exists(conn_list_fn):
            print 'Merging default connection files...'
            merge_pattern = params['mpn_bgd1_conn_fn_base']
            fn_out = params['mpn_bgd1_merged_conn_fn']
            utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)


    def plot_wij_for_iteration(self, it, gid_pairs, ax):

        pre_gids = []
        post_gids = []
        for i_, (pre_gid, post_gid) in enumerate(gid_pairs):
            wij = self.trace_buffer[it][(pre_gid, post_gid)]['wij']
            x = np.arange(wij.size)
            ax.plot(x, wij, label='%d-%d' % (pre_gid, post_gid))
            pre_gids.append(pre_gid)
            post_gids.append(post_gid)

        pre_gids = np.unique(pre_gids)
        post_gids = np.unique(post_gids)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.annotate('Pre gids: %s\nPost gids: %s' % (str(list(pre_gids)), str(list(post_gids))), \
                (xlim[0] + .1 * (xlim[1] - xlim[0]), \
                 ylim[0] + .1 * (ylim[1] - ylim[0])))

        # plot voltages
        ax.set_title('Training it = %d' % it)
        ax.set_ylabel('w_ij')
        ax.set_xlabel('Time [ms]')


    def create_voltage_plotter(self, params, key_string, it_range):
        self.VP[key_string] = VoltPlotter(params, it_range)


    def plot_spikes_raster(self, params, gids, it_range, ax, color='k'):

        # plot the pre-synaptic spike train
        fn_pre_spikes = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
        try:
            spikes_pre = self.spike_data[fn_pre_spikes] 
        except:
            self.spike_data[fn_pre_spikes] = np.loadtxt(fn_pre_spikes)
            spikes_pre = self.spike_data[fn_pre_spikes] 

        t_range = (it_range[0] * params['t_iteration'], it_range[1] * params['t_iteration'])
        spikes_it = utils.get_spiketimes_within_interval(spikes_pre, t_range[0], t_range[1])
        for gid in gids:
            spikes_pre_gid = utils.get_spiketimes(spikes_it, gid)
            nspikes = len(spikes_pre_gid)
            ax.plot(spikes_pre_gid, gid * np.ones(nspikes), 'o', markersize=5, markeredgewidth=0, color=color)
        ax.set_xlim(t_range)
        ax.set_ylim((1, params['n_exc_mpn'] + 1))
        if params['training']:
            ax.set_title('Training')
        else:
            ax.set_title('Testing')


    def plot_iteration_borders(self, params, ax, it_range):
        ylim = ax.get_ylim()

        for it in xrange(it_range[0], it_range[1]):
            t0, t1 = it * params['t_iteration'], (it + 1) * params['t_iteration']
            ax.plot((t0, t0), (ylim[0], ylim[1]), '--', c='k')
            ax.plot((t1, t1), (ylim[0], ylim[1]), '--', c='k')
#            ax.plot((t1, ylim[0]), (t1, ylim[1]), '--', c='k')


    def plot_pre_post_activity(self, params, VP_key, pre_gid, post_gid, it_range, ax1, ax2):

        # plot the pre-synaptic spike train
        fn_pre_spikes = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
        try:
            spikes_pre = self.spike_data[fn_pre_spikes] 
        except:
            self.spike_data[fn_pre_spikes] = np.loadtxt(fn_pre_spikes)
            spikes_pre = self.spike_data[fn_pre_spikes] 

        t_range = (it_range[0] * params['t_iteration'], it_range[1] * params['t_iteration'])
        spikes_it = utils.get_spiketimes_within_interval(spikes_pre, t_range[0], t_range[1])
        spikes_pre_gid = utils.get_spiketimes(spikes_it, pre_gid)
#        self.spike_pre_gid[
        nspikes = len(spikes_pre_gid)

        VP = self.VP[VP_key]
#        if VP_key == 'testing':
#            print '\n\n BBBBB', VP_key, self.VP[VP_key].loaded_files.keys()
#            print 'CCCCCCC', VP_key, VP.loaded_files.keys()
        # plot the post-synaptic membrane potential
        t_axis, volt = VP.get_trace(post_gid, fn_base=params['bg_volt_fn'])
        if len(t_axis) == 0:
            print 'No voltage trace for neuron %d has been recorder' % post_gid

        ax1.set_title('%s iteration %d-%d' % (VP_key, it_range[0], it_range[1]))
        ax1.plot(spikes_pre_gid, pre_gid * np.ones(nspikes), 'o', markersize=5, markeredgewidth=0, color='k')
        ax2.plot(t_axis, volt)
        ax1.set_ylim((1, params['n_exc_mpn'] + 1))
        ax2.set_ylim((-85, -54))
        ax2.set_ylabel('Voltage [mV]')
        ax2.set_xlabel('Time [ms]')
        ax1.set_xlim(t_range)
        ax2.set_xlim(t_range)


    def plot_voltages(self, params, VP_key, gids, ax, it_range):

        VP = self.VP[VP_key]
        for gid in gids: 
            t_axis, volt = VP.get_trace(gid, fn_base=params['bg_volt_fn'])
            if len(t_axis) == 0:
                print 'No voltage trace for neuron %d has been recorder' % gid

        t_range = (it_range[0] * params['t_iteration'], it_range[1] * params['t_iteration'])
        ax.plot(t_axis, volt)
        ax.set_ylim((-85, -54))
        ax.set_ylabel('Voltage [mV]')
        ax.set_xlabel('Time [ms]')
        ax.set_xlim(t_range)



    def set_trace_buffer(self, it, gid_pairs, traces):
        self.trace_buffer[it] = {}
        for i_, (pre_gid, post_gid) in enumerate(gid_pairs):
            self.trace_buffer[(pre_gid, post_gid)] = {}
            self.trace_buffer[it][(pre_gid, post_gid)] = {}
            self.trace_buffer[it][(pre_gid, post_gid)]['wij'] = traces[i_][0]
            self.trace_buffer[it][(pre_gid, post_gid)]['pij'] = traces[i_][4]
            self.trace_buffer[it][(pre_gid, post_gid)]['eij'] = traces[i_][7]

            self.trace_buffer[it][pre_gid] = {}
            self.trace_buffer[it][pre_gid]['pi'] = traces[i_][2]
            self.trace_buffer[it][pre_gid]['ei'] = traces[i_][5]
            self.trace_buffer[it][pre_gid]['zi'] = traces[i_][8]

            self.trace_buffer[it][post_gid] = {}
            self.trace_buffer[it][post_gid] = {}
            self.trace_buffer[it][post_gid]['bias'] = traces[i_][1]
            self.trace_buffer[it][post_gid]['pj'] = traces[i_][3]
            self.trace_buffer[it][post_gid]['ej'] = traces[i_][6]
            self.trace_buffer[it][post_gid]['zj'] = traces[i_][9]


def get_weights(pre_gids, post_gids, d):
    """
    d -- n x 3 array (src, tgt, weight)
    """

    for i_, pre_gid in enumerate(pre_gids):
        targets = utils.get_targets(d, pre_gid)
        print 'Projecting from %d' % (pre_gid), targets[:, 2].mean()

    for i_, post_gid in enumerate(post_gids):
        sources = utils.get_sources(d, post_gid)
        print 'Projecting to %d' % (post_gid), sources[:, 2].mean()



def get_weight(pre_gid, post_gid, d):
    pre_idx = (d[:, 0] == pre_gid).nonzero()[0]
    idx = (d[pre_idx, 1] == post_gid).nonzero()[0]
    if idx.size > 0:
        w = d[idx, 2]
        return w
    else:
        return 0. #None


def get_target_action(pre_gid, d, bg_cell_gids, actions_taken, iteration):
    targets = utils.get_targets(d, pre_gid)
    print 'Pre GID %d' % (pre_gid), 
    target_actions = []
    for i_, gid in enumerate(targets[:, 1]):
        action_idx = utils.map_gid_to_action(gid, bg_cell_gids)
        target_actions.append(action_idx)

    print 'Target actions:', np.unique(target_actions)
    n_tgt_actions = np.unique(target_actions).size
    tgt_weight = np.zeros(n_tgt_actions)
#    tgt_weight = np.zeros(n_tgt_actions)
    tgt_weights = {}
    n_tgts = {}
    for action in np.unique(target_actions):
        tgt_weights[action] = 0.
        n_tgts[action] = 0

    for i_ in xrange(targets[:, 2].size):
        tgt, w = targets[i_, 1], targets[i_, 2]
        action_idx = utils.map_gid_to_action(gid, bg_cell_gids)
        tgt_weights[action_idx] += w
        n_tgts[action] += 1

    for action in np.unique(target_actions):
        if tgt_weights[action] != 0.:
            w_mean = tgt_weights[action_idx] / n_tgts[action]
#            check = action == actions_taken[iteration + 1, 2]
            print 'Source %d\tTarget Action: %d\tTarget Weight.sum = %.3f\tmean: %.3f\t\tCheck: wiring %d - %d (should be)' % (pre_gid, action, tgt_weights[action], w_mean, action, actions_taken[iteration+1, 2])





if __name__ == '__main__':

    training_params = utils.load_params( os.path.abspath(sys.argv[1]) )

    colorlist = utils.get_colorlist()
    n_color = len(colorlist)
    cell_type_post = 'd2'

    DB = DebugTraces()
    DB.check_spike_files(training_params)

    TP_training = TracePlotter(training_params)
    fn_pre = training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged']
    fn_post = training_params['spiketimes_folder'] + training_params['%s_spikes_fn_merged_all' % cell_type_post]
    TP_training.load_spikes(fn_pre, fn_post)

    stim_range_global = (0, 1)
#    stim_range_global = (0, training_params['n_stim_training'])
    it_range_global = (training_params['n_iterations_per_stim'] * stim_range_global[0], training_params['n_iterations_per_stim'] * stim_range_global[1])
#    it_range_bcpnn_in_stim = [0, 7] # within the first stimulus
#    it_range_bcpnn = (training_params['n_iterations_per_stim'] * stim_range_global[0] + it_range_bcpnn_in_stim[0], training_params['n_iterations_per_stim'] * stim_range_global[1] + it_range_bcpnn_in_stim[1])
#    plot_bcpnn_iteration_pre = int(sys.argv[2]) # decides from which iteration the most-active neurons are taken 
    it_range_bcpnn = it_range_global
    print 'it_range_bcpnn:', it_range_bcpnn
    all_pre_gids = []
    all_post_gids = []

    pre_gids_in_iteration = {} # stores the most active pre nrns for each iteration
    post_gids_in_iteration = {} # stores the most active post nrns for each iteration
    post_gids = {} # stores the most active post nrns for each iteration
    all_gid_pairs = {} # stores the most active (pre, post) pairs for each iteration

    most_active_pre_gids = {}
    most_active_post_gids = {}
    n_pre = 2   # number of most active neurons per iteration 
    n_post = 2  
    for it in xrange(it_range_global[0], it_range_global[1]):
        it_range = (it, it+1)
        pre_gids, post_gids = TP_training.select_cells(n_pre=n_pre, n_post=n_post, it_range=it_range)
        all_pre_gids += list(pre_gids)
        all_post_gids += list(post_gids)
        most_active_pre_gids[it] = pre_gids
        most_active_post_gids[it] = post_gids
#        traces, gid_pairs = TP_training.compute_traces(pre_gids, post_gids, it_range)
#        DB.set_trace_buffer(it, gid_pairs, traces)
#        all_gid_pairs[it] = gid_pairs

    actions_taken = np.loadtxt(training_params['actions_taken_fn'])
    # ---------- TRAINING -----------------
    # --- get the weights between all the pre and post cells
    conn_fn = training_params['mpn_bgd1_merged_conn_fn']
    f_bg = file(training_params['bg_gids_fn'], 'r')
    bg_cell_gids = json.load(f_bg)
    d = np.loadtxt(conn_fn)
    for iteration in xrange(it_range_global[0], it_range_global[1]):
        print '------------ Iteration %d   t = [%d - %d] ------------ ' % (iteration, iteration * training_params['t_iteration'], (iteration + 1) * training_params['t_iteration'])
        get_weights(most_active_pre_gids[iteration], most_active_post_gids[iteration], d)
        for pre_gid in most_active_pre_gids[iteration]:
            get_target_action(pre_gid, d, bg_cell_gids, actions_taken, iteration)
#            for post_gid in most_active_post_gids[iteration]:
#                w = get_weight(pre_gid, post_gid, d)
#                print 'w(%d - %d) : %.3f' % (pre_gid, post_gid, w)
#    exit(1)

    # -------- PLOT SPIKES (color coded for actions) ----------------
    # --- plot the spikes from the most active cells for the different iterations
    fig_test = pylab.figure()
    ax1_training = fig_test.add_subplot(111)
    for it in xrange(it_range_global[0], it_range_global[1]):
        pre_gids = most_active_pre_gids[it]
        DB.plot_spikes_raster(training_params, pre_gids, it_range_global, ax1_training, color=colorlist[it % n_color])
        post_gids = most_active_post_gids[it]
        DB.plot_spikes_raster(training_params, post_gids, it_range_global, ax1_training, color=colorlist[it % n_color])
    DB.plot_iteration_borders(training_params, ax1_training, it_range_global)
    ylim = ax1_training.get_ylim()
    for it in xrange(it_range_global[0], it_range_global[1]):
        if not np.isnan(actions_taken[it, 2]):
            t0 = it * training_params['t_iteration']
            ax1_training.text(t0 + .2 * training_params['t_iteration'], 1.05 * ylim[1], 'action %d' % (actions_taken[it, 2]), fontsize=16, color=colorlist[it % n_color])
    ax1_training.set_xlabel('Time [ms]')
    ax1_training.set_ylabel('Visual layer GID')
    print 'You should (re-)run the training & testing to record V_m from these cells:'
    print 'gids_to_record_mpn:', list(np.unique(all_pre_gids))
    print 'gids_to_record_bg:', list(np.unique(all_post_gids))


    # ---------- BCPNN TRACES --------------------
    # choose the most active pre - post gids for one iteration
    # and plot the BCPNN traces for a certain time
    plot_bcpnn_iteration_pre = 0
    plot_bcpnn_iteration_post = 0
    assert (plot_bcpnn_iteration_pre in it_range_global), ' plot_bcpnn_iteration_pre follows global iteration numbering (not the wihtin-stimulus numbering'
    assert (plot_bcpnn_iteration_post in it_range_global), ' plot_bcpnn_iteration_post follows global iteration numbering (not the wihtin-stimulus numbering'
    pre_gids = most_active_pre_gids[plot_bcpnn_iteration_pre][:n_pre].tolist()
    post_gids = most_active_post_gids[plot_bcpnn_iteration_post][:n_post].tolist()

    all_bcpnn_traces, gid_pairs = TP_training.compute_traces(pre_gids, post_gids, it_range_bcpnn)

    # ---PLOT BCPNN TRACES
    fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    for i_ in xrange(len(all_bcpnn_traces)):
        print 'Plotting gid_pair', gid_pairs[i_]
        traces = all_bcpnn_traces[i_]
        if i_ == (len(all_bcpnn_traces) - 1):
            output_fn = training_params['figures_folder'] + 'bcpnn_traces_man%d+%d_iteration%d-%d.png' % (plot_bcpnn_iteration_pre, plot_bcpnn_iteration_post, it_range_bcpnn[0], it_range_bcpnn[1])
            info_txt = 'Iteration %d (stim %d) action %d' % (plot_bcpnn_iteration_pre, plot_bcpnn_iteration_pre / training_params['n_iterations_per_stim'], actions_taken[plot_bcpnn_iteration_post+1, 2])
            info_txt += '\nMost active nrns (pre %d, post %d): \nPre: %s\nPost: %s' % (plot_bcpnn_iteration_pre, plot_bcpnn_iteration_post, pre_gids, post_gids)
        else:
            output_fn = None
            info_txt = ''
        TP_training.plot_trace(traces, output_fn=output_fn, info_txt=info_txt, fig=fig)



    # ---------- TESTING -----------------
#    testing_params = utils.load_params(os.path.abspath(sys.argv[2]))
#    DB.check_spike_files(testing_params)
    # Plot MPN spike activity and BG voltages for selected cells 
#    fig_test = pylab.figure()
#    ax1_testing = fig_test.add_subplot(211)
#    for it in xrange(it_range_global[0], it_range_global[1]):
#        pre_gids = most_active_pre_gids[it]
#        DB.plot_spikes_raster(testing_params, pre_gids, it_range_global, ax1_testing, color=colorlist[it])
#        post_gids = most_active_post_gids[it]
#        DB.plot_spikes_raster(testing_params, post_gids, it_range_global, ax1_testing, color=colorlist[it])
#    DB.plot_iteration_borders(testing_params, ax1_testing, it_range_global)
#    ylim = ax1_testing.get_ylim()
#    for it in xrange(it_range_global[0], it_range_global[1]):
#        t0 = it * testing_params['t_iteration']
#        ax1_testing.text(t0 + .05 * testing_params['t_iteration'], ylim[0] + (.02 * (ylim[1] - ylim[0])), 'action %d' % (actions_taken[it, 2]), fontsize=16, color=colorlist[it])
#    ax1_testing.set_xlabel('Time [ms]')
#    ax1_testing.set_ylabel('Visual layer GID')

#    plot_volt_iteration = 2
#    ax2_test = fig_test.add_subplot(212)
#    DB.create_voltage_plotter(testing_params, 'testing', it_range_global)
#    plot_volt_gids = post_gids_in_iteration[plot_volt_iteration]
#    DB.plot_voltages(testing_params, 'testing', plot_volt_gids, ax2_test, (0, plot_volt_iteration+1))


    # =====================
#        DB.plot_pre_post_activity(training_params, 'training', gid_pair[0], gid_pair[1], it_range_global, ax1_train, ax2_train)
#        gid_pairs = all_gid_pairs[plot_iteration]
#        print 'Iteration %d gid_pair' % (plot_iteration), gid_pairs
#        ax1_test = fig_testing.add_subplot(211)
#        ax2_test = fig_testing.add_subplot(212)
#        fig_training = pylab.figure()
#        ax1_train = fig_training.add_subplot(211)
#        ax2_train = fig_training.add_subplot(212)
#        DB.create_voltage_plotter(testing_params, 'testing', it_range_global)
#        DB.create_voltage_plotter(training_params, 'training', it_range_global)
#        for gid_pair in gid_pairs:
#            traces, gid_pairs = TP_training.compute_traces(pre_gids, post_gids, it_range_global)
#            DB.set_trace_buffer(plot_iteration, gid_pairs, traces)
#            DB.plot_pre_post_activity(testing_params, 'testing', gid_pair[0], gid_pair[1], it_range_global, ax1_test, ax2_test)
#            DB.plot_pre_post_activity(training_params, 'training', gid_pair[0], gid_pair[1], it_range_global, ax1_train, ax2_train)

#        xlim, ylim= ax2_train.get_xlim(), ax2_train.get_ylim()
#        ax2_train.annotate('Pre gids: %s\nPost gids: %s' % (str(pre_gids_in_iteration[plot_iteration]), str(post_gids_in_iteration[plot_iteration])), \
#                (xlim[0] + .1 * (xlim[1] - xlim[0]), \
#                 ylim[0] + .1 * (ylim[1] - ylim[0])))

#        xlim, ylim= ax2_test.get_xlim(), ax2_test.get_ylim()
#        ax2_test.annotate('Pre gids: %s\nPost gids: %s' % (str(pre_gids_in_iteration[plot_iteration]), str(post_gids_in_iteration[plot_iteration])), \
#                (xlim[0] + .1 * (xlim[1] - xlim[0]), \
#                 ylim[0] + .1 * (ylim[1] - ylim[0])))

#        fig_wij = pylab.figure()
#        ax_wij = fig_wij.add_subplot(111)
#        DB.plot_wij_for_iteration(plot_iteration, gid_pairs, ax_wij)

#    DB.plot_wij_for_iteration(plot_iteration, gid_pairs, ax_wij)
    pylab.show()
