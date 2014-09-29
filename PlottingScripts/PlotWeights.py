import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import sys
import os
import utils
import re
import json
import numpy as np
import pylab
import matplotlib
import MergeSpikefiles
import simulation_parameters
import plot_conn_list_as_colormap as PlotCmap

def plot_final_weights(params, action_idx):

    P = PlotWeights(params)
    cell_type = 'd1'
    clim = (-5, 5)

    P.plot_final_weights(cell_type, clim=clim)
    P.get_weights_to_action(action_idx, cell_type)


def get_weights_versus_iterations(params, action_idx, it_range_pre, cell_type='d1'):

    P = PlotWeights(params)
    for it_ in xrange(params['n_iterations']):
        P.merge_conn_dev_files(it_, cell_type)
    utils.remove_empty_files(params['connections_folder'])

    conn_files = utils.get_connection_files(params, cell_type)
#    exit(1)
    n_conn_data = len(conn_files) # how many measurements for weights exist
    # select the presynaptic cells for which the weights are to plotted
    pre_gids = select_presynaptic_cells(params, it_range=it_range_pre)
    n_pre = len(pre_gids)
    assert (n_pre > 0), 'ERROR: No presynaptic cells could be determined due to a lack of pre-synaptic spikes. Check spike files and paths!'

    f = file(params['bg_gids_fn'], 'r')
    bg_gids = json.load(f) 
    gids_for_action = bg_gids[cell_type][action_idx]
    print 'Plotting post gids:', gids_for_action
    n_post = len(gids_for_action)

#    weight_data = np.zeros((n_pre, n_post, n_conn_data))
    weight_data = np.zeros((n_pre * n_post, n_conn_data + 2)) # + 2 to store the pre and post gid at the beginning of the row

    # load connection files for all iterations
    for (it_, conn_fn) in enumerate(conn_files):
        print 'Loading', conn_fn
        d = np.loadtxt(conn_fn)
        for (i_, pre_gid) in enumerate(pre_gids):
            for (j_, post_gid) in enumerate(gids_for_action):
                w_ij_t = utils.extract_weight_from_connection_list(d, pre_gid, post_gid)
                row_idx = i_ * n_post + j_
                weight_data[row_idx, 0] = pre_gid
                weight_data[row_idx, 1] = post_gid
                weight_data[row_idx, it_+2] = w_ij_t
#                print 'Pre %d Post %d Weight' % (pre_gid, post_gid), weight_data[row_idx, :]

    output_fn = params['data_folder'] + 'weights_action_%d_it%d-%d.dat' % (action_idx, it_range_pre[0], it_range_pre[1])
    print 'Saving weight data to:', output_fn
    np.savetxt(output_fn, weight_data)
    return output_fn

#    for it_ in xrange(params['n_iterations']):
#        fn = params['mpn_bg%s_merged_conntracking_fn_base' % cell_type] + 'it%d.txt' % (it_)
#        os.path.exists(fn):
#        try:
#            d = np.loadtxt(fn)
            # extract the weight for the action_idx


def plot_weights_versus_iterations(data_fn):
    d = np.loadtxt(data_fn)

    pre_gids = np.unique(d[:, 0])
    post_gids = np.unique(d[:, 1])

    print 'plot_weights_versus_iterations:'
    print 'pre_gids:', pre_gids
    print 'post_gids:', post_gids
    n_points = d[0, 2:].size

    colorlist = utils.get_colorlist(post_gids.size)
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for (j_, post_gid) in enumerate(post_gids):
        row_idx = (d[:, 1] == post_gid).nonzero()[0]
        for row in row_idx:
            ax.plot(xrange(n_points), d[row, 2:], c=colorlist[j_])


def get_active_actions(params, it_range=None):
    if it_range == None:
        it_range = np.array([0, params['n_iterations']])
    activity_memory = np.loadtxt(params['activity_memory_fn'])
    active_actions = set([])
    for i_ in xrange(it_range):
        new_action_idx = set(activity_memory[i_, :].nonzero()[0])
        active_actions = active_actions.union(new_action_idx)
    return active_actions




class WeightPlotter(object):
    def __init__(self, params):
        self.params = params
        self.params = params
        self.conn_data = {} 
        self.conn_data['d1'] = {} # connection lists
        self.conn_data['d2'] = {} # connection lists
        self.conn_mat = {} # connection matrix
        for it in xrange(self.params['n_iterations']):
            self.conn_data['d1'][it] = None
            self.conn_data['d2'][it] = None
        self.pre_spikes_loaded = False
        f = file(params['bg_gids_fn'], 'r')
        self.bg_gids = json.load(f) 
        self.w_ij_data = {} # w_ij_data[pre_gid][post_gid] = list of weights from pre -> post
        self.p_ij_data = {} # w_ij_data[pre_gid][post_gid] = list of weights from pre -> post
        self.p_i_data = {} # w_ij_data[pre_gid][post_gid] = list of weights from pre -> post
        self.p_j_data = {} # w_ij_data[pre_gid][post_gid] = list of weights from pre -> post

    def load_weights(self, it, tgt_celltype='d1'):
        """
        tgt_celltype -- 'd1' or 'd2'
        """
        fn = self.params['mpn_bg%s_merged_conntracking_fn_base' % tgt_celltype] + 'it%04d.txt' % it
        if self.conn_data[tgt_celltype][it] == None:
            print 'Loading:', fn
            file_size = os.path.getsize(fn)
            if file_size != 0:
                d = np.loadtxt(fn) 
                self.conn_data[tgt_celltype][it] = d


    def load_final_weights(self, cell_type='d1'):
        conn_fn = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        print 'Loading:', conn_fn
        d = np.loadtxt(conn_fn)
        return d



    def load_pre_spikes(self):
        fn_pre_spikes = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
        if not os.path.exists(fn_pre_spikes):
            utils.merge_spikes(params)
        self.pre_spikes = np.loadtxt(fn_pre_spikes)
        self.pre_spikes_loaded = True


    def select_presynaptic_cells(self, it_range=(0, 1), n_max=None):
        if not self.pre_spikes_loaded:
            self.load_pre_spikes()
        t_range = (it_range[0] * self.params['t_iteration'], it_range[1] * self.params['t_iteration'])
        spikes_filtered_by_time = utils.get_spiketimes_within_interval(self.pre_spikes, t_range[0], t_range[1])
        if n_max != None:
            (pre_gids, nspikes) = utils.get_most_active_neurons(spikes_filtered_by_time, n_max)
        else:
            (pre_gids, nspikes) = utils.get_most_active_neurons(spikes_filtered_by_time)
        assert (len(pre_gids) > 0), 'ERROR: No presynaptic cells could be determined due to a lack of pre-synaptic spikes. Check spike files and paths!'
        return pre_gids


    def plot_weights(self, tgt_cell_type, it_range_plotting, it_range_for_pre_cell_selection, action_idx):
        pre_gids = self.select_presynaptic_cells(it_range=it_range_for_pre_cell_selection)
        gids_for_action = self.bg_gids[tgt_cell_type][action_idx]

        print 'Plotting post gids:', gids_for_action
        print 'Plotting pre gids:', list(pre_gids)
        self.load_wij_data(pre_gids, gids_for_action, tgt_cell_type, it_range_plotting)

        fig = pylab.figure()
        ax = fig.add_subplot(111)

        fig = pylab.figure()
        ax_pi = fig.add_subplot(111)
        fig = pylab.figure()
        ax_pj = fig.add_subplot(111)
        fig = pylab.figure()
        ax_pij = fig.add_subplot(111)
        for pre_gid in pre_gids:
            for post_gid in gids_for_action:
                y_ = self.w_ij_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
                x_ = range(len(y_))
                ax.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))

                y_ = self.p_ij_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
                ax_pij.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))

                y_ = self.p_j_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
                ax_pj.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))

                y_ = self.p_i_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
                ax_pi.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))

#        pylab.legend()
        self.save_wij_data()
        ax.set_title('weight to %s - action %d' % (tgt_cell_type, action_idx))
        ax_pij.set_title('p_ij to %s - action %d' % (tgt_cell_type, action_idx))
        ax_pi.set_title('p_i to %s - action %d' % (tgt_cell_type, action_idx))
        ax_pj.set_title('p_j to %s - action %d' % (tgt_cell_type, action_idx))


    def load_wij_data(self, pre_gids, post_gids, tgt_cell_type, it_range):
        for it_ in xrange(params['n_iterations']):
            self.merge_conn_dev_files(it_, tgt_cell_type)
            self.load_weights(it_, tgt_cell_type)
        utils.remove_empty_files(params['connections_folder'])
        

        for pre_gid in pre_gids:
            for post_gid in post_gids:
                for it_ in xrange(params['n_iterations']):
                    if self.conn_data[tgt_cell_type][it_] != None:
                        w_ij_t = utils.extract_weight_from_connection_list(self.conn_data[tgt_cell_type][it_], pre_gid, post_gid)
                        p_i_t = utils.extract_weight_from_connection_list(self.conn_data[tgt_cell_type][it_], pre_gid, post_gid, idx=3)
                        p_j_t = utils.extract_weight_from_connection_list(self.conn_data[tgt_cell_type][it_], pre_gid, post_gid, idx=4)
                        p_ij_t = utils.extract_weight_from_connection_list(self.conn_data[tgt_cell_type][it_], pre_gid, post_gid, idx=5)
                        if not self.w_ij_data.has_key(pre_gid):
                            self.w_ij_data[pre_gid] = {}
                            self.w_ij_data[pre_gid][post_gid] = []
                            self.w_ij_data[pre_gid][post_gid].append(w_ij_t)
                        elif not self.w_ij_data[pre_gid].has_key(post_gid):
                            self.w_ij_data[pre_gid][post_gid] = []
                            self.w_ij_data[pre_gid][post_gid].append(w_ij_t)
                        else:
                            self.w_ij_data[pre_gid][post_gid].append(w_ij_t)

                        if not self.p_ij_data.has_key(pre_gid):
                            self.p_ij_data[pre_gid] = {}
                            self.p_ij_data[pre_gid][post_gid] = []
                            self.p_ij_data[pre_gid][post_gid].append(p_ij_t)
                        elif not self.p_ij_data[pre_gid].has_key(post_gid):
                            self.p_ij_data[pre_gid][post_gid] = []
                            self.p_ij_data[pre_gid][post_gid].append(p_ij_t)
                        else:
                            self.p_ij_data[pre_gid][post_gid].append(p_ij_t)

                        if not self.p_i_data.has_key(pre_gid):
                            self.p_i_data[pre_gid] = {}
                            self.p_i_data[pre_gid][post_gid] = []
                            self.p_i_data[pre_gid][post_gid].append(p_i_t)
                        elif not self.p_i_data[pre_gid].has_key(post_gid):
                            self.p_i_data[pre_gid][post_gid] = []
                            self.p_i_data[pre_gid][post_gid].append(p_i_t)
                        else:
                            self.p_i_data[pre_gid][post_gid].append(p_i_t)

                        if not self.p_j_data.has_key(pre_gid):
                            self.p_j_data[pre_gid] = {}
                            self.p_j_data[pre_gid][post_gid] = []
                            self.p_j_data[pre_gid][post_gid].append(p_j_t)
                        elif not self.p_j_data[pre_gid].has_key(post_gid):
                            self.p_j_data[pre_gid][post_gid] = []
                            self.p_j_data[pre_gid][post_gid].append(p_j_t)
                        else:
                            self.p_j_data[pre_gid][post_gid].append(p_j_t)



#        print 'debug WIJ', self.w_ij_data


    def save_wij_data(self):       

        fn_out = self.params['data_folder'] + 'weight_trace_data.json'
        print 'Saving to:', fn_out
        f = file(fn_out, 'w')
        json.dump(self.w_ij_data, f, indent=2)
        # assuming that all pre-post cell pairs have the same number of weight points
#        rnd_pre_gid = [self.wij_data.keys()[0]
#        rnd_post_gid = [self.wij_data.keys()[0][0]
#        n_ = len(self.w_ij_data[pre_gid][post_gid])
#        d_out = 
#        for 


    def merge_conn_dev_files(self, it, tgt_cell_type='d1'):
        merge_pattern = self.params['mpn_bg%s_conntracking_fn_base' % tgt_cell_type] + 'it%04d' % (it)
        fn_out = self.params['mpn_bg%s_merged_conntracking_fn_base' % tgt_cell_type] + 'it%04d.txt' % (it)
        utils.merge_and_sort_files(merge_pattern, fn_out, sort=False, verbose=False)


class PlotWeights(object):

    def __init__(self, params):
        self.params = params
        self.conn_data = {} 
        self.conn_data['d1'] = {} # connection lists
        self.conn_data['d2'] = {} # connection lists
        # self.conn_data[cell_type][iteration]
        self.conn_mat = {} # connection matrix
        for it in xrange(self.params['n_iterations']):
            self.conn_data['d1'][it] = None
            self.conn_data['d2'][it] = None


    def plot_weights_vs_iterations_old(self, src_gids, tgt_gids_type, cell_type, it_max=None):
        """
        """
        if it_max == None:
            it_max = self.params['n_iterations']

        if cell_type == 'd1':
            tgt_gids = tgt_gids_type['strD1']
        elif cell_type == 'd2':
            tgt_gids = tgt_gids_type['strD2']
        w_data = np.zeros((it_max, len(src_gids), len(tgt_gids)))
        print 'DEBUG plotting projections from', src_gids, '\n\ttgt_gids', tgt_gids

        for it in xrange(it_max):
            self.load_weights(it, cell_type)

        colorlist = utils.get_colorlist()
        linestyles = utils.get_linestyles()
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for i_, src_gid in enumerate(src_gids):
            for j_, tgt_gid in enumerate(tgt_gids):
                for it in xrange(it_max):
                    d = self.conn_data[cell_type][it]
                    src_idx = (d[:, 0] == src_gid).nonzero()[0]
                    tgt_idx = (d[src_idx, 1] == tgt_gid).nonzero()[0]
                    if tgt_idx.size == 0:
                        w_data[it, i_, j_] = 0
                    else:
#                        print 'debug it', it
#                        print 'debug', d[tgt_idx, 2], tgt_idx, src_gid, tgt_gid
#                        print 'debug2', src_idx, d[src_idx, 1], d[src_idx, 1] == tgt_gid
                        w_data[it, i_, j_] = d[tgt_idx, 2]
#                print 'w_data', w_data[:, i_, j_], src_gid, tgt_gid
                ax.plot(t_axis, w_data[:, i_, j_], c=colorlist[i_ % len(colorlist)], ls=linestyles[j_ % len(linestyles)], lw=3)

        ax.set_title('Target cell_type: %s' % cell_type)
        ax.set_ylabel('Weight')
        ax.set_xlabel('Training iteration')



    def plot_final_weights(self, cell_type, clim=None):
        conn_fn = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        conn_mat = PlotCmap.plot_conn_list(conn_fn, params=self.params, clim=clim)
#def plot_conn_list(conn_list_fn, params=None, clim=None, src_cell_type=None):
        self.conn_mat[cell_type] = conn_mat
        print 'min max weights', conn_mat.min(), conn_mat.max()


    def get_weights_to_action(self, action_number, cell_type):
        f = file(self.params['bg_gids_fn'], 'r')
        bg_gids = json.load(f)
        gids_ = bg_gids[cell_type][action_number]
        gid_offset = bg_gids[cell_type][0][0]

        weights = np.zeros(len(gids_))

        for tgt in xrange(len(gids_)):
            w_ = self.conn_mat[cell_type][:, tgt]
            w_mean = w_.mean()
            w_std = w_.std()

#            print 'tgt %d \tmean %.3f +- %.3f\t' % (tgt, w_mean, w_std), w_
        





def compare_training_and_test(sysargv):
    training_folder = os.path.abspath(sysargv[1]) 
    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_param_tool = simulation_parameters.global_parameters(params_fn=training_params_fn)
    training_params = training_param_tool.params

    test_folder = os.path.abspath(sysargv[2]) 
    test_param_fn = os.path.abspath(test_folder) + '/Parameters/simulation_parameters.json'
    f = file(test_param_fn, 'r')
    test_params = json.load(f)

    bg_gids_to_check, incorrect_iterations = utils.compare_actions_taken(training_params, test_params)
    print 'bg_gids_to_check', bg_gids_to_check

    output_fn = 'bg_gids_to_check.json'
    f = file(output_fn, 'w')
    json.dump(bg_gids_to_check, f, indent=0)


    # merge spike files
    utils.merge_and_sort_files(training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn'], \
            training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged'], sort=False)

    # find MPN cells that spiked during the incorrect iterations
    mpn_gids = {}
    spike_data = np.loadtxt(training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged'])

    # get spikes within incorrect iterations
    for i_, it in enumerate(incorrect_iterations):
        t0 = it * training_params['t_iteration']
        t1 = (it + 1) * training_params['t_iteration']
        spikes_in_iteration = utils.get_spiketimes_within_interval(spike_data, t0, t1)
#        print 'MPN spikes_in_iteration %d' % it, spikes_in_iteration
        mpn_gids[it] = np.unique(spikes_in_iteration[:, 0].astype(np.int))
        print 'MPN spikes_in_iteration %d' % it, spikes_in_iteration, mpn_gids[it]

    iteration = incorrect_iterations[0]
    print 'Iteration', iteration
    print 'Source gids', mpn_gids[iteration]
    P = PlotWeights(training_params)
    P.plot_weights_vs_iterations_old(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d1', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d1.png'
    pylab.savefig(output_fig)
    P.plot_weights_vs_iterations_old(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d2', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d2.png'
    pylab.savefig(output_fig)



class PlotWeights(object):

    def __init__(self, params):
        self.params = params
        self.conn_data = {} 
        self.conn_data['d1'] = {} # connection lists
        self.conn_data['d2'] = {} # connection lists
        # self.conn_data[cell_type][iteration]
        self.conn_mat = {} # connection matrix
        for it in xrange(self.params['n_iterations']):
            self.conn_data['d1'][it] = None
            self.conn_data['d2'][it] = None


    def plot_weights_vs_iterations_old(self, src_gids, tgt_gids_type, cell_type, it_max=None):
        """
        """
        if it_max == None:
            it_max = self.params['n_iterations']

        if cell_type == 'd1':
            tgt_gids = tgt_gids_type['strD1']
        elif cell_type == 'd2':
            tgt_gids = tgt_gids_type['strD2']
        w_data = np.zeros((it_max, len(src_gids), len(tgt_gids)))
        print 'DEBUG plotting projections from', src_gids, '\n\ttgt_gids', tgt_gids

        for it in xrange(it_max):
            self.load_weights(it, cell_type)

        colorlist = utils.get_colorlist()
        linestyles = utils.get_linestyles()
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for i_, src_gid in enumerate(src_gids):
            for j_, tgt_gid in enumerate(tgt_gids):
                for it in xrange(it_max):
                    d = self.conn_data[cell_type][it]
                    src_idx = (d[:, 0] == src_gid).nonzero()[0]
                    tgt_idx = (d[src_idx, 1] == tgt_gid).nonzero()[0]
                    if tgt_idx.size == 0:
                        w_data[it, i_, j_] = 0
                    else:
#                        print 'debug it', it
#                        print 'debug', d[tgt_idx, 2], tgt_idx, src_gid, tgt_gid
#                        print 'debug2', src_idx, d[src_idx, 1], d[src_idx, 1] == tgt_gid
                        w_data[it, i_, j_] = d[tgt_idx, 2]
#                print 'w_data', w_data[:, i_, j_], src_gid, tgt_gid
                ax.plot(t_axis, w_data[:, i_, j_], c=colorlist[i_ % len(colorlist)], ls=linestyles[j_ % len(linestyles)], lw=3)

        ax.set_title('Target cell_type: %s' % cell_type)
        ax.set_ylabel('Weight')
        ax.set_xlabel('Training iteration')



    def plot_final_weights(self, cell_type, clim=None):
        conn_fn = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        conn_mat = PlotCmap.plot_conn_list(conn_fn, params=self.params, clim=clim)
#def plot_conn_list(conn_list_fn, params=None, clim=None, src_cell_type=None):
        self.conn_mat[cell_type] = conn_mat
        print 'min max weights', conn_mat.min(), conn_mat.max()


    def get_weights_to_action(self, action_number, cell_type):
        f = file(self.params['bg_gids_fn'], 'r')
        bg_gids = json.load(f)
        gids_ = bg_gids[cell_type][action_number]
        gid_offset = bg_gids[cell_type][0][0]

        weights = np.zeros(len(gids_))

        for tgt in xrange(len(gids_)):
            w_ = self.conn_mat[cell_type][:, tgt]
            w_mean = w_.mean()
            w_std = w_.std()

#            print 'tgt %d \tmean %.3f +- %.3f\t' % (tgt, w_mean, w_std), w_
        





def compare_training_and_test(sysargv):
    training_folder = os.path.abspath(sysargv[1]) 
    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_param_tool = simulation_parameters.global_parameters(params_fn=training_params_fn)
    training_params = training_param_tool.params

    test_folder = os.path.abspath(sysargv[2]) 
    test_param_fn = os.path.abspath(test_folder) + '/Parameters/simulation_parameters.json'
    f = file(test_param_fn, 'r')
    test_params = json.load(f)

    bg_gids_to_check, incorrect_iterations = utils.compare_actions_taken(training_params, test_params)
    print 'bg_gids_to_check', bg_gids_to_check

    output_fn = 'bg_gids_to_check.json'
    f = file(output_fn, 'w')
    json.dump(bg_gids_to_check, f, indent=0)


    # merge spike files
    utils.merge_and_sort_files(training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn'], \
            training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged'], sort=False)

    # find MPN cells that spiked during the incorrect iterations
    mpn_gids = {}
    spike_data = np.loadtxt(training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged'])

    # get spikes within incorrect iterations
    for i_, it in enumerate(incorrect_iterations):
        t0 = it * training_params['t_iteration']
        t1 = (it + 1) * training_params['t_iteration']
        spikes_in_iteration = utils.get_spiketimes_within_interval(spike_data, t0, t1)
#        print 'MPN spikes_in_iteration %d' % it, spikes_in_iteration
        mpn_gids[it] = np.unique(spikes_in_iteration[:, 0].astype(np.int))
        print 'MPN spikes_in_iteration %d' % it, spikes_in_iteration, mpn_gids[it]

    iteration = incorrect_iterations[0]
    print 'Iteration', iteration
    print 'Source gids', mpn_gids[iteration]
    P = PlotWeights(training_params)
    P.plot_weights_vs_iterations_old(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d1', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d1.png'
    pylab.savefig(output_fig)
    P.plot_weights_vs_iterations_old(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d2', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d2.png'
    pylab.savefig(output_fig)



class PlotWeights(object):

    def __init__(self, params):
        self.params = params
        self.conn_data = {} 
        self.conn_data['d1'] = {} # connection lists
        self.conn_data['d2'] = {} # connection lists
        # self.conn_data[cell_type][iteration]
        self.conn_mat = {} # connection matrix
        for it in xrange(self.params['n_iterations']):
            self.conn_data['d1'][it] = None
            self.conn_data['d2'][it] = None


    def plot_weights_vs_iterations_old(self, src_gids, tgt_gids_type, cell_type, it_max=None):
        """
        """
        if it_max == None:
            it_max = self.params['n_iterations']

        if cell_type == 'd1':
            tgt_gids = tgt_gids_type['strD1']
        elif cell_type == 'd2':
            tgt_gids = tgt_gids_type['strD2']
        w_data = np.zeros((it_max, len(src_gids), len(tgt_gids)))
        print 'DEBUG plotting projections from', src_gids, '\n\ttgt_gids', tgt_gids

        for it in xrange(it_max):
            self.load_weights(it, cell_type)

        colorlist = utils.get_colorlist()
        linestyles = utils.get_linestyles()
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for i_, src_gid in enumerate(src_gids):
            for j_, tgt_gid in enumerate(tgt_gids):
                for it in xrange(it_max):
                    d = self.conn_data[cell_type][it]
                    src_idx = (d[:, 0] == src_gid).nonzero()[0]
                    tgt_idx = (d[src_idx, 1] == tgt_gid).nonzero()[0]
                    if tgt_idx.size == 0:
                        w_data[it, i_, j_] = 0
                    else:
#                        print 'debug it', it
#                        print 'debug', d[tgt_idx, 2], tgt_idx, src_gid, tgt_gid
#                        print 'debug2', src_idx, d[src_idx, 1], d[src_idx, 1] == tgt_gid
                        w_data[it, i_, j_] = d[tgt_idx, 2]
#                print 'w_data', w_data[:, i_, j_], src_gid, tgt_gid
                ax.plot(t_axis, w_data[:, i_, j_], c=colorlist[i_ % len(colorlist)], ls=linestyles[j_ % len(linestyles)], lw=3)

        ax.set_title('Target cell_type: %s' % cell_type)
        ax.set_ylabel('Weight')
        ax.set_xlabel('Training iteration')



    def plot_final_weights(self, cell_type, clim=None):
        conn_fn = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        conn_mat = PlotCmap.plot_conn_list(conn_fn, params=self.params, clim=clim)
#def plot_conn_list(conn_list_fn, params=None, clim=None, src_cell_type=None):
        self.conn_mat[cell_type] = conn_mat
        print 'min max weights', conn_mat.min(), conn_mat.max()


    def get_weights_to_action(self, action_number, cell_type):
        f = file(self.params['bg_gids_fn'], 'r')
        bg_gids = json.load(f)
        gids_ = bg_gids[cell_type][action_number]
        gid_offset = bg_gids[cell_type][0][0]

        weights = np.zeros(len(gids_))

        for tgt in xrange(len(gids_)):
            w_ = self.conn_mat[cell_type][:, tgt]
            w_mean = w_.mean()
            w_std = w_.std()

#            print 'tgt %d \tmean %.3f +- %.3f\t' % (tgt, w_mean, w_std), w_
        





def compare_training_and_test(sysargv):
    training_folder = os.path.abspath(sysargv[1]) 
    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_param_tool = simulation_parameters.global_parameters(params_fn=training_params_fn)
    training_params = training_param_tool.params

    test_folder = os.path.abspath(sysargv[2]) 
    test_param_fn = os.path.abspath(test_folder) + '/Parameters/simulation_parameters.json'
    f = file(test_param_fn, 'r')
    test_params = json.load(f)

    bg_gids_to_check, incorrect_iterations = utils.compare_actions_taken(training_params, test_params)
    print 'bg_gids_to_check', bg_gids_to_check

    output_fn = 'bg_gids_to_check.json'
    f = file(output_fn, 'w')
    json.dump(bg_gids_to_check, f, indent=0)


    # merge spike files
    utils.merge_and_sort_files(training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn'], \
            training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged'], sort=False)

    # find MPN cells that spiked during the incorrect iterations
    mpn_gids = {}
    spike_data = np.loadtxt(training_params['spiketimes_folder'] + training_params['mpn_exc_spikes_fn_merged'])

    # get spikes within incorrect iterations
    for i_, it in enumerate(incorrect_iterations):
        t0 = it * training_params['t_iteration']
        t1 = (it + 1) * training_params['t_iteration']
        spikes_in_iteration = utils.get_spiketimes_within_interval(spike_data, t0, t1)
#        print 'MPN spikes_in_iteration %d' % it, spikes_in_iteration
        mpn_gids[it] = np.unique(spikes_in_iteration[:, 0].astype(np.int))
        print 'MPN spikes_in_iteration %d' % it, spikes_in_iteration, mpn_gids[it]

    iteration = incorrect_iterations[0]
    print 'Iteration', iteration
    print 'Source gids', mpn_gids[iteration]
    P = PlotWeights(training_params)
    P.plot_weights_vs_iterations_old(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d1', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d1.png'
    pylab.savefig(output_fig)
    P.plot_weights_vs_iterations_old(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d2', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d2.png'
    pylab.savefig(output_fig)



#    for it in xrange(iterations):
if __name__ == '__main__':

    if len(sys.argv) == 2:
        param_fn = sys.argv[1]
        params = utils.load_params(param_fn)
#    elif len(sys.argv) == 3:
#        compare_training_and_test(sys.argv)
    else:
        import simulation_parameters
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params


    it_range_plotting = (0, params['n_iterations'])
    it_range_pre_cell_selection = (0, 1) # determines iteration range to determine the presynaptic cells
    action_idx = 0

    WP = WeightPlotter(params)
    WP.plot_weights('d1', it_range_plotting, it_range_pre_cell_selection, action_idx)

    WP.plot_weights('d2', it_range_plotting, it_range_pre_cell_selection, action_idx)


#    data_fn = get_weights_versus_iterations(params, action_idx, it_range_pre, cell_type='d1')
#    data_fn = params['data_folder'] + 'weights_action_%d_it%d-%d.dat' % (action_idx, it_range_pre[0], it_range_pre[1])
#    plot_weights_versus_iterations(data_fn)

#    plot_final_weights(params, action_idx)

    pylab.show()

#    iterations = (0, 2)
