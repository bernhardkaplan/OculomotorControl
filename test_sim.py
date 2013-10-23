
import numpy as np
import nest

nest.SetKernelStatus({'data_path': 'tmp', 'resolution': .1, 'overwrite_files':True})

n_exc = 3
pop = nest.Create('iaf_psc_exp_multisynapse', n_exc)
stimulus = nest.Create('spike_generator', n_exc)

w_input_exc = 50.
nest.CopyModel('static_synapse', 'input_exc_0', \
        {'weight': w_input_exc, 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
nest.CopyModel('static_synapse', 'input_exc_1', \
        {'weight': w_input_exc, 'receptor_type': 1})


voltmeter = nest.Create('voltmeter')
nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
nest.ConvergentConnect(voltmeter, pop)


exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
nest.ConvergentConnect(pop, exc_spike_recorder)

n_intervals = 3
t_current = 0
t_integrate = 100
for interval in xrange(n_intervals):
    n_spikes = np.random.randint(0, 10)
    for unit in xrange(n_exc): # create a spike train for the respective unit
        n_spikes = 5
        spike_times = np.around(np.random.rand(n_spikes) * t_integrate + t_current, decimals=1)
        spike_times.sort()
        nest.SetStatus([stimulus[unit]], {'spike_times' : spike_times})
        nest.Connect([stimulus[unit]], [pop[unit]], model='input_exc_0')
        print 'interval, unit, spike_times', interval, unit, spike_times
    t_current += t_integrate
    nest.Simulate(t_integrate)
