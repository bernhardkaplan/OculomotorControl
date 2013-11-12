import nest

class CreateConnections(object):

    def __init__(self, params, comm=None):
        
        self.params = params

        nest.CopyModel('static_synapse', 'mpn_bg_exc', \
                {'weight': self.params['w_exc_mpn_bg'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])
        self.comm = comm
        if comm != None:
            self.pc_id = comm.rank
            self.n_proc = comm.size


    def connect_mt_to_bg(self, src_net, tgt_net):
        """
        The NEST simulation should run for some pre-fixed time
        Keyword arguments:
        src_net, tgt_net -- the source and the target network

        """

        for nactions in range(self.params['n_actions']):
            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d1_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD1[nactions], model=self.params['synapse_d1_MT_BG'])

            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d2_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD2[nactions], model=self.params['synapse_d2_MT_BG'])



    def get_weights(self, src_pop, tgt_pop):
        """
        After training get the weights between the MPN state layer and the BG action layer
        """

        print 'Writing weights to files...'
        D1_conns = ''
        D2_conns = ''
        for nactions in range(self.params['n_actions']):
            print 'action %d' % nactions

            conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD1[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    w = cp[0]['weight'] 
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        D1_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])


            conns = nest.GetConnections(src_pop.exc_pop, tgt_pop.strD2[nactions]) # get the list of connections stored on the current MPI node
            if conns != None:
                for c in conns:
                    cp = nest.GetStatus([c])  # retrieve the dictionary for this connection
                    w = cp[0]['weight'] 
                    if (cp[0]['synapse_model'] == 'bcpnn_synapse'):
                        D2_conns += '%d\t%d\t%.4e\n' % (cp[0]['source'], cp[0]['target'], cp[0]['weight'])

        fn_out = self.params['mpn_bgd1_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D1_f = file(fn_out, 'w')
        D1_f.write(D1_conns)
        D1_f.close()

        fn_out = self.params['mpn_bgd2_conn_fn_base'] + '%d.txt' % (self.pc_id)
        print 'Writing connections to:', fn_out
        D2_f = file(fn_out, 'w')
        D2_f.write(D2_conns)
        D2_f.close()



    def get_connection_kernel(self, src_gid):
        return 0
