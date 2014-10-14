import nest
import numpy as np
import os
import utils
import time
import json

class CreateConnections(object):

    def __init__(self, params, comm=None):
        
        self.params = params

        print nest.Models()
#        nest.CopyModel('bcpnn_synapse', 'adfadsf', params=self.params['params_synapse_d1_MT_BG'])
        nest.CopyModel('bcpnn_synapse', self.params['synapse_d1_MT_BG'], params=self.params['params_synapse_d1_MT_BG'])
        nest.CopyModel('bcpnn_synapse', self.params['synapse_d2_MT_BG'], params=self.params['params_synapse_d2_MT_BG'])

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
            nest.SetDefaults(self.params['synapse_d1_MT_BG'], params=self.params['params_synapse_d1_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD1[nactions], model=self.params['synapse_d1_MT_BG'])
            if self.params['with_d2']:
                nest.SetDefaults(self.params['synapse_d2_MT_BG'], params=self.params['params_synapse_d2_MT_BG'])
                nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD2[nactions], model=self.params['synapse_d2_MT_BG'])
        if self.comm != None:
            self.comm.Barrier()



    def merge_connection_files(self, training_params, test_params=None):

        def merge_for_tgt_cell_type(cell_type):
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

        def merge_for_weight_tracking(cell_type):
            for it in xrange(training_params['n_iterations']):
                fn_merged = training_params['mpn_bg%s_merged_conntracking_fn_base' % cell_type] + 'it%d.txt' % (it)
                if not os.path.exists(fn_merged):
                    # merge the connection files
                    merge_pattern = training_params['mpn_bg%s_conntracking_fn_base' % cell_type] + 'it%d_' % it
                    utils.merge_and_sort_files(merge_pattern, fn_merged, sort=True)

        if self.pc_id == 0:
            merge_for_tgt_cell_type('d1')
            if training_params['with_d2']:
                merge_for_tgt_cell_type('d2')

            merge_pattern = training_params['d1_d1_conn_fn_base']
            fn_out = training_params['d1_d1_merged_conn_fn']
            utils.merge_and_sort_files(merge_pattern, fn_out, sort=True)
        if self.comm != None:
            self.comm.barrier()

        if training_params['weight_tracking']:
            # Merge the _dev files recorded for tracking the weights
            if self.pc_id == 0:
                merge_for_weight_tracking('d1')
                if training_params['with_d2']:
                    merge_for_weight_tracking('d2')
        if self.comm != None:
            self.comm.barrier()


    def connect_d1_after_training(self, BG, training_params, testing_params, model='static_synapse', bcpnn_params=None):

        fn = training_params['d1_d1_merged_conn_fn']
        print 'CreateConnections.connect_d1_after_training loads', fn
        file_size = os.path.getsize(fn)
        if file_size == 0:
            utils.merge_and_sort_files(training_params['d1_d1_conn_fn_base'], training_params['d1_d1_merged_conn_fn'])

        d = np.loadtxt(fn)
        srcs = list(d[:, 0].astype(np.int))
        tgts = list(d[:, 1].astype(np.int))
        weights = np.zeros(d[:, 2].size)
        neg_idx = np.nonzero(d[:, 2] < 0)[0]
        pos_idx = np.nonzero(d[:, 2] > 0)[0]
        weights[neg_idx] = d[neg_idx, 2] * testing_params['gain_d1_d1']
        weights[pos_idx] = d[pos_idx, 2] * testing_params['gain_d1_d1']
        weights = list(weights)
        delays = list(np.ones(d[:, 0].size * testing_params['delay_d1_d1']))
        if bcpnn_params == None:
            nest.Connect(srcs, tgts, weights, delays, model=model)
#        else:
#            nest.Connect(srcs, tgts, weights, delays, params=[bcpnn_params for i in xrange(d[:, 2].size)], model=model )
    


    def connect_mt_to_bg_RBL(self, mpn_net, bg_net, training_params, test_params, target='d1', model='static_synapse'):
        """
        Connects the sensor layer (motion-prediction network, MPN) to the Basal Ganglia 
        based on the weights found in conn_folder
        """
        if self.comm != None:
            self.comm.Barrier()

        self.merge_connection_files(training_params, test_params)
        print 'Loading MPN - BG %s connections from: %s' % (target, training_params['mpn_bg%s_merged_conn_fn' % target])
        tgt_path = test_params['connections_folder'] + 'merged_mpn_bg_%s_connections_preTraining.txt' % target
        cmd = 'cp %s %s' % (training_params['mpn_bg%s_merged_conn_fn' % target], tgt_path)
        if self.pc_id == 0:
            os.system(cmd)
        mpn_bg_conn_list = np.loadtxt(training_params['mpn_bg%s_merged_conn_fn' % target])
        n_lines = mpn_bg_conn_list[:, 0].size 
        w = mpn_bg_conn_list[:, 2]
        pi = test_params['bcpnn_init_pi']
        pj = test_params['bcpnn_init_pi']
        w *= self.params['gain_MT_%s' % target]
#        pij = pi * pj * np.exp(w)
        valid_idx = np.nonzero(np.abs(w) > self.params['weight_threshold'])[0]
        srcs = list(mpn_bg_conn_list[valid_idx, 0].astype(np.int))
        tgts = list(mpn_bg_conn_list[valid_idx, 1].astype(np.int))
        weights = list(w[valid_idx])
        pij = pi * pj * np.exp(w[valid_idx])

        delays = list(np.ones(len(weights)) * self.params['mpn_bg_delay'])
        param_dict_list = [test_params['params_synapse_%s_MT_BG' % target] for i_ in xrange(valid_idx.size)]
        for i_ in xrange(valid_idx.size):
#            print 'debug',param_dict_list[i_], param_dict_list[i_]['p_i']
            param_dict_list[i_]['p_i'] = pi
            param_dict_list[i_]['p_j'] = pj
            param_dict_list[i_]['p_ij'] = pij[i_]
#        param_dict = [ {'p_i' : pi[i_], 'p_j': pj[i_], 'p_ij': pij[i_], 'weight': weights[i_], 'delay': delays[i_]} for i_ in xrange(valid_idx.size)]
            nest.Connect([srcs[i_]], [tgts[i_]], param_dict_list[i_], model=model)
#        nest.Connect(srcs, tgts, weights, delays, model=model)

        # set the pi, pj, traces
#        nest.SetStatus(nest.GetConnections(srcs, tgts, 





    def connect_mt_to_bg_after_training(self, mpn_net, bg_net, training_params, test_params, model='static_synapse', debug=False):
        """
        Connects the sensor layer (motion-prediction network, MPN) to the Basal Ganglia 
        based on the weights found in conn_folder
        """
        self.merge_connection_files(training_params, test_params)
        if self.comm != None:
            self.comm.Barrier()
        print 'Loading MPN - BG D1 connections from:', training_params['mpn_bgd1_merged_conn_fn']
        mpn_d1_conn_list = np.loadtxt(training_params['mpn_bgd1_merged_conn_fn'])
        n_lines = mpn_d1_conn_list[:, 0].size 

        w = mpn_d1_conn_list[:, 2]
        w *= self.params['gain_MT_d1']
        valid_idx = np.nonzero(np.abs(w) > self.params['weight_threshold'])[0]
        srcs = list(mpn_d1_conn_list[valid_idx, 0].astype(np.int))
        tgts = list(mpn_d1_conn_list[valid_idx, 1].astype(np.int))
        weights = list(w[valid_idx])
        delays = list(np.ones(len(weights)) * self.params['mpn_bg_delay'])
        nest.Connect(srcs, tgts, weights, delays, model=model)
        if debug:
            output_array_d1 = np.zeros((len(weights), 3))
            output_array_d1[:, 0] = srcs
            output_array_d1[:, 1] = tgts
            output_array_d1[:, 2] = weights
            mpn_d1_debug_fn = test_params['mpn_bgd1_merged_conn_fn'].rsplit('.txt')[0] + '_debug.txt'
            print 'Saving the realized connections to %s' % mpn_d1_debug_fn
            np.savetxt(mpn_d1_debug_fn, output_array_d1)

        if training_params['with_d2'] and test_params['with_d2']:
            print 'Loading MPN - BG D2 connections from:', training_params['mpn_bgd2_merged_conn_fn']
            mpn_d2_conn_list = np.loadtxt(training_params['mpn_bgd2_merged_conn_fn'])
            w = mpn_d2_conn_list[:, 2]
            w *= self.params['gain_MT_d2']
            valid_idx = np.nonzero(np.abs(w) > self.params['weight_threshold'])[0]
            srcs = list(mpn_d2_conn_list[valid_idx, 0].astype(np.int))
            tgts = list(mpn_d2_conn_list[valid_idx, 1].astype(np.int))
            weights = list(w[valid_idx])
            delays = list(np.ones(len(weights)) * self.params['mpn_bg_delay'])
            nest.Connect(srcs, tgts, weights, delays, model=model)
            if debug:
                output_array_d2 = np.zeros((len(weights), 3))
                output_array_d2[:, 0] = srcs
                output_array_d2[:, 1] = tgts
                output_array_d2[:, 2] = weights
                mpn_d2_debug_fn = test_params['mpn_bgd2_merged_conn_fn'].rsplit('.txt')[0] + '_debug.txt'
                print 'Saving the realized connections to %s' % mpn_d2_debug_fn
                np.savetxt(mpn_d2_debug_fn, output_array_d2)


    def clip_weight(self, w, clip_weights, thresh_and_abs):
        """
        clip_weights -- boolean; if False, do nothing and return w
        thresh_and_abs -- (float, boolean), float gives the threshold, and the boolean decides if the absolute value should be taken
                        i.e. if boolean == True: negative weights are allowed
        """
        if not clip_weights:
            return w
        else:
            if thresh_and_abs[1]: # allow negative weights
                if abs(w) > thresh_and_abs[0]:
                    return w
                else:
                    return None
            else: # allow only positive weights
                if w > thresh_and_abs[0]:
                    return w
                else:
                    return None

    def debug_connections(self, tgt_pop, model='static_synapse'):

        debug_txt = ''
        for i_action in xrange(self.params['n_actions']):
            conns = nest.GetConnections(target=tgt_pop[i_action])
            if conns != None:# and model == 'static_synapse':
                for i_, c in enumerate(conns):
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    debug_txt += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])
#            elif conns != None and model != 'static_synapse':
#                print '\n\nWTF\n\n'
#                exit(1)
        return debug_txt


    def debug_mpn_connections(self, pop): 
        debug_txt = ''
        output_fn = 'delme_mpn_as_source_wD1_%.1f_wD2_%.1f_%d.txt' % (self.params['gain_MT_d1'], self.params['gain_MT_d2'], self.pc_id)

        conns = nest.GetConnections(source=pop)
        if conns != None:
            for i_, c in enumerate(conns):
                cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                debug_txt += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])
#                print 'debug c', c, cp
        f = file(output_fn, 'w')
        f.write(debug_txt)
        f.flush()
        f.close()



    def get_weights(self, src_pop, tgt_pop, iteration=None, model='bcpnn_synapse'):
        """
        After training get the weights between the MPN state layer and the BG action layer
        """

        print 'Writing weights to files...'
        D1_conns = ''
        bias_d1 = {}
        for nactions in xrange(self.params['n_actions']):
            print 'CreateConnections.get_weights action %d' % nactions, 'iteration:', iteration
            conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD1[nactions], synapse_model=self.params['synapse_d1_MT_BG']) # get the list of connections stored on the current MPI node
#            print 'DEBUG src_pop.exc_pop:', src_pop.exc_pop
#            print 'DEBUG tgt_pop:', tgt_pop.strD1[nactions]
#            print 'DEBUG %d n_conns: %d' % (self.pc_id, len(conns))
#            exit(1)
#            print 'DEBUG conns', conns
            if conns != None and model != 'static_synapse':
                for i_, c in enumerate(conns):
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    pi = cp[0]['p_i']
                    pj = cp[0]['p_j']
                    pij = cp[0]['p_ij']
                    w = np.log(pij / (pi * pj))
                    D1_conns += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij)
#                        w_ = self.clip_weight(w, self.params['clip_weights_mpn_d1'], self.params['weight_threshold_abstract_mpn_d1'])
#                        if w_:
#                    D1_conns += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w_, pi, pj, pij)
                    bias_d1[cp[0]['target']] = cp[0]['bias']
            elif conns != None and model == 'static_synapse':
                for i_, c in enumerate(conns):
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    D1_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])

        if iteration == None:
            fn_out = self.params['mpn_bgd1_conn_fn_base'] + '%d.txt' % (self.pc_id)
        else:
            fn_out = self.params['mpn_bgd1_conntracking_fn_base'] + 'it%04d_%d.txt' % (iteration, self.pc_id)
        bias_d1_f = file(self.params['bias_d1_fn_base'] + 'pc%d.json' % self.pc_id, 'w')
        json.dump(bias_d1, bias_d1_f, indent=0)
        print 'Writing MPN - D1 connections to:', fn_out
        D1_f = file(fn_out, 'w')
        D1_f.write(D1_conns)
        D1_f.close()

        if self.params['with_d2']:
            D2_conns = ''
            bias_d2 = {}
            for nactions in xrange(self.params['n_actions']):
                conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD2[nactions], synapse_model=self.params['synapse_d2_MT_BG']) # get the list of connections stored on the current MPI node
                if conns != None and model != 'static_synapse':
                    for c in conns:
                        cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        D2_conns += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij)
                elif conns != None and model == 'static_synapse':
                    for i_, c in enumerate(conns):
                        cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                        D2_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])
#                            w_ = self.clip_weight(w, self.params['clip_weights_mpn_d2'], self.params['weight_threshold_abstract_mpn_d2'])
#                            if w_:
#                                D2_conns += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w_, pi, pj, pij)
#                                bias_d2[cp[0]['target']] = cp[0]['bias']

            bias_d2_f = file(self.params['bias_d2_fn_base'] + 'pc%d.json' % self.pc_id, 'w')
            json.dump(bias_d2, bias_d2_f, indent=0)
            if iteration == None:
                fn_out = self.params['mpn_bgd2_conn_fn_base'] + '%d.txt' % (self.pc_id)
            else:
                fn_out = self.params['mpn_bgd2_conntracking_fn_base'] + 'it%04d_%d.txt' % (iteration, self.pc_id)
            print 'Writing MPN - D2 connections to:', fn_out
            D2_f = file(fn_out, 'w')
            D2_f.write(D2_conns)
            D2_f.close()


    def get_d1_d1_weights(self, BG):

        D1_conns = ''
        for i_ in xrange(self.params['n_actions']):
            for j_ in xrange(self.params['n_actions']):
                print 'CreateConnections.get_d1_d1_weights action %d - %d '% (i_, j_)
                conns = nest.GetConnections(BG.strD1[i_], BG.strD1[j_], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        w_ = self.clip_weight(w, self.params['clip_weights_d1_d1'], self.params['weight_threshold_abstract_d1_d1'])
                        if w_: # ignore the positive weights between d1 neurons
                            D1_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w_)

        fn_out = self.params['d1_d1_conn_fn_base'] + '%d.txt' % (self.pc_id)
        D1_f = file(fn_out, 'w')
        D1_f.write(D1_conns)
        D1_f.close()



    def get_d2_d2_weights(self, BG):
        D2_conns = ''
        for i_ in xrange(self.params['n_actions']):
            for j_ in xrange(self.params['n_actions']):
                print 'CreateConnections.get_d2_d2_weights action %d - %d '% (i_, j_)
                conns = nest.GetConnections(BG.strD2[i_], BG.strD2[j_], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
#                print 'DEBUG conns:', conns
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        pi = cp[0]['p_i']
                        pj = cp[0]['p_j']
                        pij = cp[0]['p_ij']
                        w = np.log(pij / (pi * pj))
                        w_ = self.clip_weight(w, self.params['clip_weights_d2_d2'], self.params['weight_threshold_abstract_d2_d2'])
                        if w_: # ignore the positive weights between d2 neurons
                            D2_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w_)

        fn_out = self.params['d2_d2_conn_fn_base'] + '%d.txt' % (self.pc_id)
        D2_f = file(fn_out, 'w')
        D2_f.write(D2_conns)
        D2_f.close()


