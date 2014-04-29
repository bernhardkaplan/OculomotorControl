import nest
import numpy as np
import os
import utils
import time
import json

class CreateConnections(object):

    def __init__(self, params, comm=None):
        
        self.params = params

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])
        self.comm = comm
        if comm != None:
            self.pc_id = comm.rank
            self.n_proc = comm.size
        else:
            self.pc_id = 0
            self.n_proc = 1


    def connect_mt_to_bg(self, src_net, tgt_net):
        """
        The NEST simulation should run for some pre-fixed time
        Keyword arguments:
        src_net, tgt_net -- the source and the target network

        """

        for nactions in xrange(self.params['n_actions']):
            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d1_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD1[nactions], model=self.params['synapse_d1_MT_BG'])

            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d2_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD2[nactions], model=self.params['synapse_d2_MT_BG'])




    def merge_connection_files(self, training_params, test_params=None):

        # merge the final weight files
        if self.pc_id == 0:

            for cell_type in ['d1', 'd2']:
                if not os.path.exists(training_params['mpn_bg%s_merged_conn_fn' % cell_type]):
                    # merge the connection files
                    merge_pattern = training_params['mpn_bg%s_conn_fn_base' % cell_type]
                    fn_out = training_params['mpn_bg%s_merged_conn_fn' % cell_type]
                    utils.merge_and_sort_files(merge_pattern, fn_out, sort=True)

                if test_params != None: 
                    bias_fns = utils.find_files(training_params['connections_folder'], 'bias_%s_pc' % cell_type)
                    all_bias = {}
                    for fn in bias_fns:
                        bias_fn = training_params['connections_folder'] + fn
                        f = file(bias_fn, 'r')
                        bias_data = json.load(f)
                        all_bias.update(bias_data)
                    f_out = file(test_params['bias_%s_merged_fn' % cell_type], 'w')
                    json.dump(all_bias, f_out, indent=0)

        if self.comm != None:
            self.comm.barrier()

        if training_params['weight_tracking']:
            # Merge the _dev files recorded for tracking the weights
            if self.pc_id == 0:
                for it in xrange(self.training_params['n_iterations']):
                    fn_merged = self.training_params['mpn_bgd1_merged_conntracking_fn_base'] + 'it%d.txt' % (it)
                    if not os.path.exists(fn_merged):
                        # merge the connection files
                        merge_pattern = training_params['mpn_bgd1_conntracking_fn_base'] + 'it%d_' % it
                        utils.merge_and_sort_files(merge_pattern, fn_merged, sort=True)
                for it in xrange(self.training_params['n_iterations']):
                    fn_merged = self.training_params['mpn_bgd2_merged_conntracking_fn_base'] + 'it%d.txt' % (it)
                    if not os.path.exists(fn_merged):
                        # merge the connection files
                        merge_pattern = training_params['mpn_bgd2_conntracking_fn_base'] + 'it%d_' % it
                        utils.merge_and_sort_files(merge_pattern, fn_merged, sort=True)
        if self.comm != None:
            self.comm.barrier()


    def connect_mt_to_bg_after_training(self, mpn_net, bg_net, training_params, test_params):
        """
        Connects the sensor layer (motion-prediction network, MPN) to the Basal Ganglia 
        based on the weights found in conn_folder
        """
        self.merge_connection_files(training_params, test_params)
        print 'Loading MPN - BG D1 connections from:', training_params['mpn_bgd1_merged_conn_fn']
        mpn_d1_conn_list = np.loadtxt(training_params['mpn_bgd1_merged_conn_fn'])
        n_lines = mpn_d1_conn_list[:, 0].size 

        w = mpn_d1_conn_list[:, 2]
        w *= self.params['mpn_d1_weight_amplification']
        valid_idx = np.nonzero(np.abs(w) > self.params['weight_threshold'])[0]
        srcs = list(mpn_d1_conn_list[valid_idx, 0].astype(np.int))
        tgts = list(mpn_d1_conn_list[valid_idx, 1].astype(np.int))
        weights = list(w[valid_idx])
        delays = list(np.ones(len(weights)) * self.params['mpn_bg_delay'])
        nest.Connect(srcs, tgts, weights, delays)
        output_array_d1 = np.zeros((len(weights), 3))
        output_array_d1[:, 0] = srcs
        output_array_d1[:, 1] = tgts
        output_array_d1[:, 2] = weights
        mpn_d1_debug_fn = test_params['mpn_bgd1_merged_conn_fn'].rsplit('.txt')[0] + '_debug.txt'
        print 'Saving the realized connections to %s' % mpn_d1_debug_fn
        np.savetxt(mpn_d1_debug_fn, output_array_d1)

        print 'Loading MPN - BG D2 connections from:', training_params['mpn_bgd2_merged_conn_fn']
        mpn_d2_conn_list = np.loadtxt(training_params['mpn_bgd2_merged_conn_fn'])
        w = mpn_d2_conn_list[:, 2]
        w *= self.params['mpn_d2_weight_amplification']
        valid_idx = np.nonzero(np.abs(w) > self.params['weight_threshold'])[0]
        srcs = list(mpn_d2_conn_list[valid_idx, 0].astype(np.int))
        tgts = list(mpn_d2_conn_list[valid_idx, 1].astype(np.int))
        weights = list(w[valid_idx])
        delays = list(np.ones(len(weights)) * self.params['mpn_bg_delay'])
        nest.Connect(srcs, tgts, weights, delays)
        output_array_d2 = np.zeros((len(weights), 3))
        output_array_d2[:, 0] = srcs
        output_array_d2[:, 1] = tgts
        output_array_d2[:, 2] = weights
        mpn_d2_debug_fn = test_params['mpn_bgd2_merged_conn_fn'].rsplit('.txt')[0] + '_debug.txt'
        print 'Saving the realized connections to %s' % mpn_d2_debug_fn
        np.savetxt(mpn_d2_debug_fn, output_array_d2)


    def clip_weight(self, w):
        if not self.params['clip_weights']:
            return w
        else:
            if self.params['weight_threshold_abstract'][1]:
                if abs(w) > self.params['weight_threshold_abstract'][0]:
                    return w
                else:
                    return None
            else:
                if w > self.params['weight_threshold_abstract'][0]:
                    return w
                else:
                    return None



    def get_weights(self, src_pop, tgt_pop, iteration=None):
        """
        After training get the weights between the MPN state layer and the BG action layer
        """

        print 'Writing weights to files...'
        D1_conns = ''
        D2_conns = ''
        bias_d1 = {}
        bias_d2 = {}
        for nactions in xrange(self.params['n_actions']):
            print 'CreateConnections.get_weights action %d' % nactions, 'iteration:', iteration
            conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD1[nactions], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
            if conns != None:
                for i_, c in enumerate(conns):
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        w_ = self.clip_weight(w)
                        if w_:
                            D1_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w_)
                            bias_d1[cp[0]['target']] = cp[0]['bias']

            conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD2[nactions], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        w_ = self.clip_weight(w)
                        if w_:
                            D2_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w_)
                            bias_d2[cp[0]['target']] = cp[0]['bias']

        if iteration == None:
            fn_out = self.params['mpn_bgd1_conn_fn_base'] + '%d.txt' % (self.pc_id)
        else:
            fn_out = self.params['mpn_bgd1_conntracking_fn_base'] + 'it%d_%d.txt' % (iteration, self.pc_id)

        bias_d1_f = file(self.params['bias_d1_fn_base'] + 'pc%d.json' % self.pc_id, 'w')
        json.dump(bias_d1, bias_d1_f, indent=0)
        bias_d2_f = file(self.params['bias_d2_fn_base'] + 'pc%d.json' % self.pc_id, 'w')
        json.dump(bias_d2, bias_d2_f, indent=0)

        print 'Writing connections to:', fn_out
        D1_f = file(fn_out, 'w')
        D1_f.write(D1_conns)
        D1_f.close()

        if iteration == None:
            fn_out = self.params['mpn_bgd2_conn_fn_base'] + '%d.txt' % (self.pc_id)
        else:
            fn_out = self.params['mpn_bgd2_conntracking_fn_base'] + 'it%d_%d.txt' % (iteration, self.pc_id)

        print 'Writing connections to:', fn_out
        D2_f = file(fn_out, 'w')
        D2_f.write(D2_conns)
        D2_f.close()


    def set_pc_id(self, pc_id):
        self.pc_id = pc_id
