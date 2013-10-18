import nest

class CreateConnections(object):

    def __init__(self, params):
        
        self.params = params

        nest.CopyModel('static_synapse', 'mpn_bg_exc', \
                {'weight': self.params['w_exc_mpn_bg'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc

        nest.SetDefaults(self.params['bcpnn'], params=self.params['param_bcpnn'])


    def connect_mt_to_bg(self, src_net, tgt_net):
        """
        The NEST simulation should run for some pre-fixed time
        Keyword arguments:
        src_net, tgt_net -- the source and the target network

        """

        for nactions in range(self.params['n_actions']):
            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d1_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD1[nactions], model=self.params['synapse_d1_MT_BG'], weight=self.params['w_ei_mpn'])

            nest.SetDefaults(self.params['bcpnn'], params=self.params['params_synapse_d2_MT_BG'])
            nest.ConvergentConnect(src_net.exc_pop, tgt_net.strD2[nactions], model=self.params['synapse_d2_MT_BG'])


    def get_connection_kernel(self, src_gid):
        return 0
