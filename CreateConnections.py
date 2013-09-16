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
        for src_ in xrange(self.params['n_states']):
            for tgt_ in xrange(self.params['n_states']):
                src_pop = src_net.list_of_populations[src_]
                tgt_pop = tgt_net.list_of_populations[tgt_]
                nest.ConvergentConnect(src_pop, tgt_pop, model='mpn_bg_exc')

