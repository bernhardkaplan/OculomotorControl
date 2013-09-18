import nest

class CreateConnections(object):

    def __init__(self, params):
        
        self.params = params

        nest.CopyModel('static_synapse', 'mpn_bg_exc', \
                {'weight': self.params['w_exc_mpn_bg'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc

    def connect_mt_to_bg(self, src_net, tgt_net):
        """
        The NEST simulation should run for some pre-fixed time
        Keyword arguments:
        src_net, tgt_net -- the source and the target network
        """
        for src_nrn in xrange(self.params['n_exc_mpn']):
            weights = self.get_connection_kernel(src_nrn)
            nest.DivergentConnect([src_net.exc_pop[src_nrn]], tgt_net.exc_pop, weights, delays, model='bcpnn_synapse')
#            for tgt_ in xrange(self.params['n_states']):
#                src_pop = src_net.list_of_populations[src_]
#                tgt_pop = tgt_net.list_of_populations[tgt_]
#                nest.ConvergentConnect(src_pop, tgt_pop, model='mpn_bg_exc')

    def get_connection_kernel(self, src_gid):
        return 0
