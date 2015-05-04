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


nest.SetKernelStatus({'overwrite_files': True, \
        'resolution' : 0.1})

GP = simulation_parameters.global_parameters()
params = GP.params

n_cells = 2 # per population
src_pop = nest.Create(params['neuron_model_mpn'], n_cells, params=params['cell_params_exc_mpn'])
tgt_pop = nest.Create(params['neuron_model_mpn'], n_cells, params=params['cell_params_exc_mpn'])


try:
    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
    nest.Install('pt_module')
except:
    nest.Install('pt_module')

bcpnn_init = 0.0001
tau_i = 20.
tau_j = 1.
tau_e = .1
#tau_e = 10000.
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
nest.SetStatus([trigger_spikes_src[1]], {'spike_times' : input_spike_times_src})
nest.SetStatus([trigger_spikes_tgt[1]], {'spike_times' : input_spike_times_tgt})

#nest.SetStatus([trigger_spikes_src[0]], {'spike_times' : input_spike_times_src})
print 'debug', input_spike_times_tgt



voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.1})
nest.SetStatus(voltmeter, [{"to_file": True, "withtime": True, 'label' : 'delme_volt_after_'}])
nest.ConvergentConnect(voltmeter, src_pop)
nest.ConvergentConnect(voltmeter, tgt_pop)
exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'delme_spikes_after_'})
nest.ConvergentConnect(src_pop, exc_spike_recorder)
nest.ConvergentConnect(tgt_pop, exc_spike_recorder)


conn_fn = 'delme_conns.txt'
mpn_bg_conn_list = np.loadtxt(conn_fn)
n_lines = mpn_bg_conn_list[:, 0].size 
srcs = mpn_bg_conn_list[:, 0]
tgts = mpn_bg_conn_list[:, 1]
w = mpn_bg_conn_list[:, 2]
pi = mpn_bg_conn_list[:, 3]
pj = mpn_bg_conn_list[:, 4]
pij = mpn_bg_conn_list[:, 5]

ei = mpn_bg_conn_list[:, 6]
ej = mpn_bg_conn_list[:, 7]
eij = mpn_bg_conn_list[:, 8]
zi = mpn_bg_conn_list[:, 9]
zj = mpn_bg_conn_list[:, 10]

delays = list(np.ones(w.size) * params['mpn_bg_delay'])
target = 'd1'
param_dict_list = [deepcopy(params['params_synapse_%s_MT_BG' % target]) for i_ in xrange(n_lines)]
model = params['synapse_%s_MT_BG' % target]
#fantasy = 666.

"""
for i_ in xrange(n_lines):
    param_dict_list[i_]['p_i'] = pi[i_]
    param_dict_list[i_]['p_j'] = pj[i_]
#    param_dict_list[i_]['p_j'] = fantasy
    param_dict_list[i_]['p_ij'] = pij[i_]
#    param_dict_list[i_]['p_ij'] = fantasy
    param_dict_list[i_]['weight'] = 0. #weights[i_]

#    print 'debug setting p_ij for src %d -> tgt %d to: %.3e' % (srcs[i_], tgts[i_], param_dict_list[i_]['p_ij']), fantasy
#    nest.Connect([int(srcs[i_])], [int(tgts[i_])], param_dict_list[i_], model=model)
#    nest.SetStatus(nest.GetConnections([int(srcs[i_])], [int(tgts[i_])]), {'p_i': pi[i_], 'p_j' : pj[i_], 'p_ij': fantasy})
#    print 'after setting the connection for %d - %d pij is:' % (srcs[i_], tgts[i_])
#    conns = nest.GetConnections([int(srcs[i_])], [int(tgts[i_])])
#    cp = nest.GetStatus(conns)
#    print 'pij:', cp[0]['p_ij']

    print 'debug setting p_ij for src %d -> tgt %d to: %.3e' % (srcs[i_], tgts[i_], param_dict_list[i_]['p_ij']), pij[i_]
    nest.Connect([int(srcs[i_])], [int(tgts[i_])], param_dict_list[i_], model=model)
    nest.SetStatus(nest.GetConnections([int(srcs[i_])], [int(tgts[i_])]), {'p_i': pi[i_], 'p_j' : pj[i_], 'p_ij': pij[i_]})
    conns = nest.GetConnections([int(srcs[i_])], [int(tgts[i_])])
    cp = nest.GetStatus(conns)
    print 'after drawing the connection for %d - %d pij is: %.4e' % (srcs[i_], tgts[i_], cp[0]['p_ij'])
"""

for i_ in xrange(n_lines):
    param_dict_list[i_]['z_i'] = zi[i_]
    param_dict_list[i_]['z_j'] = zj[i_]
    param_dict_list[i_]['e_i'] = ei[i_]
    param_dict_list[i_]['e_j'] = ej[i_]
    param_dict_list[i_]['e_ij'] = eij[i_]
    param_dict_list[i_]['p_i'] = pi[i_]
    param_dict_list[i_]['p_j'] = pj[i_]
    param_dict_list[i_]['p_ij'] = pij[i_]
    param_dict_list[i_]['weight'] = 0. #weights[i_]

    print 'debug setting p_ij for src %d -> tgt %d to: %.3e' % (srcs[i_], tgts[i_], param_dict_list[i_]['p_ij']), pij[i_]
    nest.Connect([int(srcs[i_])], [int(tgts[i_])], param_dict_list[i_], model=model)

#    nest.SetStatus(nest.GetConnections([int(srcs[i_])], [int(tgts[i_])]), {'p_i': pi[i_], 'p_j' : pj[i_], 'p_ij': pij[i_]})

    nest.SetStatus(nest.GetConnections([int(srcs[i_])], [int(tgts[i_])]), {'p_i': pi[i_], 'p_j' : pj[i_], 'p_ij': pij[i_], \
            'e_i': ei[i_], 'e_j': ej[i_], 'e_ij': eij[i_], 'z_i':zi[i_], 'z_j': zj[i_]})

    conns = nest.GetConnections([int(srcs[i_])], [int(tgts[i_])])
    cp = nest.GetStatus(conns)
    print 'after drawing the connection for %d - %d pij is: %.4e' % (srcs[i_], tgts[i_], cp[0]['p_ij'])





#nest.ConvergentConnect(src_pop, tgt_pop, model=params['synapse_d1_MT_BG'])


nest.Simulate(40.)
print '\nAfter simulate:'

print 'END:'
conn_txt = ''
#for gid_pre in src_pop:
#    for gid_post in tgt_pop:
#        print 'debug', type(gid_pre)
#        conns = nest.GetConnections([gid_pre], [gid_post], synapse_model=model)
#        cp = nest.GetStatus(conns)
#        print 'src tgt', gid_pre, gid_post, conns, cp[0]['source'], cp[0]['target']
#        pi = cp[0]['p_i']
#        pj = cp[0]['p_j']
#        pij = cp[0]['p_ij']
#        w = np.log(pij / (pi * pj))
#        conn_txt  += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij)
#        print '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij)

#        pi = cp[0]['p_i']
#        pj = cp[0]['p_j']
#        pij = cp[0]['p_ij']
#        ei = cp[0]['e_i']
#        ej = cp[0]['e_j']
#        eij = cp[0]['e_ij']
#        zi = cp[0]['z_i']
#        zj = cp[0]['z_j']
#        w = np.log(pij / (pi * pj))
#        conn_txt  += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij, \
#                ei, ej, eij, zi, zj)





for i_ in xrange(n_lines):
#    param_dict_list[i_]['p_i'] = pi[i_]
#    param_dict_list[i_]['p_j'] = pj[i_]
#    param_dict_list[i_]['p_ij'] = pij[i_]
#    param_dict_list[i_]['weight'] = 0. #weights[i_]
    conns = nest.GetConnections([int(srcs[i_])], [int(tgts[i_])])
    cp = nest.GetStatus(conns)
    print 'after the simulation connection for %d - %d pij is: %.4e' % (srcs[i_], tgts[i_], cp[0]['p_ij'])
    pi = cp[0]['p_i']
    pj = cp[0]['p_j']
    pij = cp[0]['p_ij']
    print 'debug pij', pij
    ei = cp[0]['e_i']
    ej = cp[0]['e_j']
    eij = cp[0]['e_ij']
    zi = cp[0]['z_i']
    zj = cp[0]['z_j']
    w = np.log(pij / (pi * pj))
    conn_txt  += '%d\t%d\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n' % (cp[0]['source'], cp[0]['target'], w, pi, pj, pij, \
            ei, ej, eij, zi, zj)
#    print 'cp:', cp



#print conns
output_fn = 'delme_conns_after_loading.txt'
print 'output_fn', output_fn
f = file(output_fn, 'w')
f.write(conn_txt)
f.flush()
f.close()

nest.PrintNetwork()
