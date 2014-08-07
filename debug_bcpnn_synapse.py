import numpy as np 
import nest

try:
    nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
    nest.Install('pt_module')
except:
    nest.Install('pt_module')

nest.SetKernelStatus({'overwrite_files': True, \
        'resolution' : 0.1})


nrns = nest.Create('iaf_cond_exp_bias', 2)

source = nest.Create('spike_generator', 1)
spike_times = [50., 100., 150.]

nest.SetStatus(source, {'spike_times' : spike_times})

tau_i = 5.
tau_j = 5.
tau_e = 5.
tau_p = 250. 
epsilon = 1. / (150. * tau_p)

bcpnn_init = 0.01

p_i = bcpnn_init
p_j = bcpnn_init
weight = 8.92
p_ij = np.exp(weight) * p_i * p_j

syn_params  = {'p_i': p_i, 'p_j': p_j, 'p_ij': p_ij, 'gain': 1.0, 'K': 0., \
        'fmax': 150., 'epsilon': epsilon, 'delay':1.0, \
        'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}


nest.Connect(source, [nrns[0]], weight, 1., model='static_synapse')
nest.Connect(source, [nrns[1]], syn_params, model='bcpnn_synapse')

exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'delme_spikes'})
nest.ConvergentConnect(nrns, exc_spike_recorder)

voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.1})
nest.SetStatus(voltmeter, [{"to_file": True, "withtime": True, 'label' : 'delme_volt'}])
nest.ConvergentConnect(voltmeter, nrns)

nest.Simulate(200.)
