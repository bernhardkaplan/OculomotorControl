import sys
import os
import json
import simulation_parameters
import CreateConnections
import nest
import numpy as np
import time
import os
import utils
from copy import deepcopy

try:
    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
    nest.Install('pt_module')
except:
    nest.Install('pt_module')

nest.SetKernelStatus({'overwrite_files': True, \
        'resolution' : 0.1})

GP = simulation_parameters.global_parameters()
params = GP.params

n_cells = 2 # per population
src_pop = nest.Create(params['neuron_model_mpn'], n_cells, params=params['cell_params_exc_mpn'])
tgt_pop = nest.Create(params['neuron_model_mpn'], n_cells, params=params['cell_params_exc_mpn'])

bcpnn_init = 0.01
tau_i = 20.
tau_j = 1.
tau_e = .1
tau_p = 10000.
gain = 0.
K = 1.
epsilon = 1. / (params['fmax'] * tau_p)
params['params_synapse_d1_MT_BG'] = {'p_i': bcpnn_init, 'p_j': bcpnn_init, 'p_ij': bcpnn_init**2, 'gain': gain, 'K': K, \
        'fmax': params['fmax'], 'epsilon': epsilon, 'delay':1.0, \
        'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
nest.CopyModel('bcpnn_synapse', params['synapse_d1_MT_BG'], params=params['params_synapse_d1_MT_BG'])

w_trigger_spikes_mpn = 30.
nest.CopyModel('static_synapse', 'trigger_synapse', {'weight': w_trigger_spikes_mpn, 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
trigger_spikes_src = nest.Create('spike_generator', n_cells)
trigger_spikes_tgt = nest.Create('spike_generator', n_cells)

################
# INPUT -> SRC
################
nest.Connect([trigger_spikes_src[0]], [src_pop[0]], model='trigger_synapse')
nest.Connect([trigger_spikes_src[1]], [src_pop[1]], model='trigger_synapse')
################
# INPUT -> TGT
################
nest.Connect([trigger_spikes_tgt[0]], [tgt_pop[0]], model='trigger_synapse')
nest.Connect([trigger_spikes_tgt[1]], [tgt_pop[1]], model='trigger_synapse')

# CREATE INPUT
nspikes_src = 4
nspikes_tgt = 4
dt = 10.
input_spike_times_src = [10. + i * dt for i in xrange(nspikes_src)]
input_spike_times_tgt = [10. + i * dt for i in xrange(nspikes_tgt)]
nest.SetStatus([trigger_spikes_src[0]], {'spike_times' : input_spike_times_src})
nest.SetStatus([trigger_spikes_tgt[0]], {'spike_times' : input_spike_times_tgt})

# CONNECT RECORDING
voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.1})
nest.SetStatus(voltmeter, [{"to_file": True, "withtime": True, 'label' : 'delme_volt'}])
nest.ConvergentConnect(voltmeter, src_pop)
nest.ConvergentConnect(voltmeter, tgt_pop)
exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'delme_spikes'})
nest.ConvergentConnect(src_pop, exc_spike_recorder)
nest.ConvergentConnect(tgt_pop, exc_spike_recorder)

#####################
# CONNECT SRC -> TGT
#####################
nest.ConvergentConnect(src_pop, tgt_pop, model=params['synapse_d1_MT_BG'])

#bcpnn_init = 0.005
#p_i = bcpnn_init
#p_j = bcpnn_init
#p_ij = bcpnn_init
#for gid_pre in src_pop:
#    for gid_post in tgt_pop:
#        nest.SetStatus(nest.GetConnections([gid_pre], [gid_post]), {'p_i' : p_i, 'p_j': p_j, 'p_ij': p_ij})

#        nest.SetStatus(nest.GetConnections([gid_pre], [gid_post], synapse_model=params['synapse_d1_MT_BG'] )

nest.Simulate(200.)


conn_txt = ''
#for gid_pre in src_pop:
#    for gid_post in tgt_pop:
#        conns = nest.GetConnections([gid_pre], [gid_post], synapse_model=params['synapse_d1_MT_BG'])
#        print 'src tgt', gid_pre, gid_post, conns
#        cp = nest.GetStatus(conns)
#        print 'status:', cp
#        pi = cp[0]['p_i']
#        pj = cp[0]['p_j']
#        pij = cp[0]['p_ij']
#        w = np.log(pij / (pi * pj))
#        conn_txt  += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij)

for gid_pre in src_pop:
    for gid_post in tgt_pop:
        conns = nest.GetConnections([gid_pre], [gid_post], synapse_model=params['synapse_d1_MT_BG'])
        print 'src tgt', gid_pre, gid_post, conns
        cp = nest.GetStatus(conns)
#        print 'status:', cp
        pi = cp[0]['p_i']
        pj = cp[0]['p_j']
        pij = cp[0]['p_ij']
        ei = cp[0]['e_i']
        ej = cp[0]['e_j']
        eij = cp[0]['e_ij']
        zi = cp[0]['z_i']
        zj = cp[0]['z_j']
        w = np.log(pij / (pi * pj))
        conn_txt  += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij, \
                ei, ej, eij, zi, zj)

output_fn = 'delme_conns.txt'
print 'output_fn', output_fn
f = file(output_fn, 'w')
f.write(conn_txt)
f.flush()
f.close()

nest.PrintNetwork()
