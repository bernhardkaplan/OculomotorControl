import numpy as np 
import nest

try:
#    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
    nest.Install('pt_module')
except:
    nest.Install('pt_module')

nest.SetKernelStatus({'overwrite_files': True, \
        'resolution' : 0.1})

cell_params = {'C_m': 250.0, 'E_L': -70.0, 'E_ex': 0.0, \
    'E_in': -80.0, 'I_e': 0.0, 'V_m': -70.0, 'V_reset': -80.0, 'V_th': -60.0, \
    'g_L': 16.6667, 't_ref': 2.0, 'tau_syn_ex': 5.0, 'tau_syn_in': 5.0}

nrns = nest.Create('iaf_cond_exp_bias', 2, params=cell_params)
voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.1})
nest.SetStatus(voltmeter, [{"to_file": True, "withtime": True, 'label' : 'delme_volt'}])
nest.ConvergentConnect(voltmeter, nrns)
exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'delme_spikes'})
nest.ConvergentConnect(nrns, exc_spike_recorder)

source_0 = nest.Create('spike_generator', 1)
source_1 = nest.Create('spike_generator', 1)

# pre synaptic spikes
#spike_times_0 = np.array([50.])# 100., 300., 350., 400.])
#nest.SetStatus(source_0, {'spike_times' : spike_times_0})
# post synaptic spikes 
spike_times_1 = np.array([])# 51., 101., 201., 221., 231., 241., 300., 350., 400.])
nest.SetStatus(source_1, {'spike_times' : spike_times_1})

weight = 15.
nest.Connect(source_0, [nrns[0]], weight, 1., model='static_synapse')
nest.Connect(source_1, [nrns[1]], weight, 1., model='static_synapse')

# neuron 0 -> neuron 1
tau_i = 5.
tau_j = 5.
tau_e = 300.
tau_p = 50000.
fmax = 200.
epsilon = 1. / (fmax * tau_p)
bcpnn_init = 0.01
p_i = bcpnn_init
p_j = bcpnn_init
weight = 0.
p_ij = np.exp(weight) * p_i * p_j
syn_params  = {'p_i': p_i, 'p_j': p_j, 'p_ij': p_ij, 'gain': 1.0, 'K': 0.0, \
        'fmax': fmax, 'epsilon': epsilon, 'delay':1.0, \
        'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
print 'syn_params init:', syn_params
nest.Connect([nrns[0]], [nrns[1]], syn_params, model='bcpnn_synapse')


# 0 - 150
# Kappa OFF + correlated activity
#nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': 0., 'gain': 1.})
#nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': 0., 'gain': 0., 't_k': nest.GetKernelStatus()['time']})
nest.Simulate(150.)

# pre synaptic spikes
#spike_times_0 = np.array([160., 170.])
#nest.SetStatus(source_0, {'spike_times' : spike_times_0})
# post synaptic spikes 
spike_times_1 = np.array([165., 175.])
nest.SetStatus(source_1, {'spike_times' : spike_times_1})


# 150 - 250: no presynaptic activity
# Kappa ON: 150 - 250 only post-synaptic spikes
#nest.Simulate(20.)
#nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': 1., 'gain': 0.})

nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': .7, 'gain': 0., 't_k': nest.GetKernelStatus()['time']})
nest.Simulate(50.)

spike_times_0 = np.array([210., 220.])
nest.SetStatus(source_0, {'spike_times' : spike_times_0})

nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': .3, 'gain': 0., 't_k': nest.GetKernelStatus()['time']})
nest.Simulate(50.)
#nest.Simulate(80.)

# 250 - 350
# Kappa OFF + correlated activity
#nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': 1., 'gain': 0.})
nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': 0., 'gain': 0., 't_k': nest.GetKernelStatus()['time']})
nest.Simulate(100.)

conns = nest.GetConnections([nrns[0]], [nrns[1]], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
#print 'conns', conns
if len(conns) > 0:
#    print 'len conns', len(conns)
    print 'conns[0]', conns[0][0]
    cp = nest.GetStatus([conns[0]])  # retrieve the dictionary for this connection
    print 'cp', cp
    pi = cp[0]['p_i']
    pj = cp[0]['p_j']
    pij = cp[0]['p_ij']
    w = np.log(pij / (pi * pj))
    print 'weight after simulation:', w

all_events = nest.GetStatus(exc_spike_recorder)[0]['events']
print 'all_events', all_events

