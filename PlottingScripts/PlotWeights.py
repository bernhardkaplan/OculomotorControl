import os, sys, inspect # use this if you want to include modules from a subforder
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

    def load_weights(self, it, cell_type='d1'):
        """
        cell_type -- 'd1' or 'd2'
        """
        fn = self.params['mpn_bg%s_merged_conntracking_fn_base' % cell_type] + 'it%04d.txt' % it
        if self.conn_data[cell_type][it] == None:
            print 'Loading:', fn
            file_size = os.path.getsize(fn)
            if file_size != 0:
                d = np.loadtxt(fn) 
                self.conn_data[cell_type][it] = d

    def load_final_weights(self, cell_type):
        conn_fn = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        print 'Loading:', conn_fn
        if os.path.getsize(conn_fn):
            d = np.loadtxt(conn_fn)
            self.conn_data[cell_type][self.params['n_iterations']-1] = d



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

        print 'Plotting post gids (action_idx=%d):' % (action_idx), gids_for_action
        print 'Plotting pre gids:', list(pre_gids)
        self.load_wij_data(pre_gids, gids_for_action, tgt_cell_type, it_range_plotting)

        fig = pylab.figure()
        ax = fig.add_subplot(111)

#        fig = pylab.figure()
#        ax_pi = fig.add_subplot(111)
#        fig = pylab.figure()
#        ax_pj = fig.add_subplot(111)
#        fig = pylab.figure()
#        ax_pij = fig.add_subplot(111)
        for pre_gid in pre_gids:
            for post_gid in gids_for_action:
                y_ = self.w_ij_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
                x_ = range(len(y_))
                ax.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))

#                y_ = self.p_ij_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
#                ax_pij.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))
#                y_ = self.p_j_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
#                ax_pj.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))
#                y_ = self.p_i_data[pre_gid][post_gid][it_range_plotting[0]:it_range_plotting[1]]
#                ax_pi.plot(x_, y_, label='%d - %d' % (pre_gid, post_gid))

#        pylab.legend()
        self.save_wij_data()
        ax.set_title('weight to %s - action %d' % (tgt_cell_type, action_idx))
#        ax_pij.set_title('p_ij to %s - action %d' % (tgt_cell_type, action_idx))
#        ax_pi.set_title('p_i to %s - action %d' % (tgt_cell_type, action_idx))
#        ax_pj.set_title('p_j to %s - action %d' % (tgt_cell_type, action_idx))


    def load_wij_data(self, pre_gids, post_gids, tgt_cell_type, it_range):
        self.merge_final_weights(tgt_cell_type)
        self.load_final_weights(tgt_cell_type)
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

    def merge_final_weights(self, cell_type='d1'):
        fn_out = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        merge_pattern = self.params['mpn_bg%s_conn_fn_base' % cell_type]
        utils.merge_and_sort_files(merge_pattern, fn_out, sort=False, verbose=False)




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
    action_idx = 16

    WP = WeightPlotter(params)
    WP.plot_weights('d1', it_range_plotting, it_range_pre_cell_selection, action_idx)

    WP.plot_weights('d2', it_range_plotting, it_range_pre_cell_selection, action_idx)


#    data_fn = get_weights_versus_iterations(params, action_idx, it_range_pre, cell_type='d1')
#    data_fn = params['data_folder'] + 'weights_action_%d_it%d-%d.dat' % (action_idx, it_range_pre[0], it_range_pre[1])
#    plot_weights_versus_iterations(data_fn)

#    plot_final_weights(params, action_idx)

    pylab.show()

#    iterations = (0, 2)
