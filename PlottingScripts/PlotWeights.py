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


class PlotWeights(object):

    def __init__(self, params):
        self.params = params
        self.conn_data = {} 
        self.conn_data['d1'] = {} # connection lists
        self.conn_data['d2'] = {} # connection lists
        self.conn_mat = {} # connection matrix
        for it in xrange(self.params['n_iterations']):
            self.conn_data['d1'][it] = None
            self.conn_data['d2'][it] = None


    def plot_weights_vs_iteration(self, src_gids, tgt_gids_type, cell_type, it_max=None):
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
        t_axis = range(it_max)
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


    def load_weights(self, it, tgt_celltype='d1'):
        """
        tgt_celltype -- 'd1' or 'd2'
        """
        fn = self.params['mpn_bg%s_merged_conntracking_fn_base' % tgt_celltype] + 'it%d.txt' % it
        if self.conn_data[tgt_celltype][it] == None:
            print 'Loading:', fn
            d = np.loadtxt(fn) 
            self.conn_data[tgt_celltype][it] = d


    def load_final_weights(self, cell_type='d1'):
        conn_fn = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        print 'Loading:', conn_fn
        d = np.loadtxt(conn_fn)
        return d



    def merge_conn_dev_files(self, it, celltype='d1'):
        merge_pattern = self.params['mpn_bg%s_conntracking_fn_base' % tgt_celltype] + 'it%d' % (it)
        conn_list_merged = self.params['mpn_bg%s_merged_conntracking_fn' % tgt_celltype] + 'it%d'
        utils.merge_and_sort_files(merge_pattern, fn_out, sort=True)


    def plot_final_weights(self, cell_type, clim=None):
        conn_fn = self.params['mpn_bg%s_merged_conn_fn' % cell_type]
        conn_mat = PlotCmap.plot_conn_list(self.params, conn_list_fn=conn_fn, clim=clim)
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
    P.plot_weights_vs_iteration(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d1', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d1.png'
    pylab.savefig(output_fig)
    P.plot_weights_vs_iteration(mpn_gids[iteration], bg_gids_to_check[iteration], cell_type='d2', it_max=15)
    output_fig = training_params['figures_folder'] + 'weight_traces_sample_d2.png'
    pylab.savefig(output_fig)



def plot_final_weights(params, action_idx):

    P = PlotWeights(params)
    cell_type = 'd1'
    clim = (-5, 5)
    P.plot_final_weights(cell_type, clim=clim)
    P.get_weights_to_action(action_idx, cell_type)


if __name__ == '__main__':

    if len(sys.argv) == 2:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)

    elif len(sys.argv) == 3:
        compare_training_and_test(sys.argv)
    else:
        import simulation_parameters
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    action_idx = 2
    plot_final_weights(params, action_idx)
    pylab.show()

#    iterations = (0, 2)
#    for it in xrange(iterations):

