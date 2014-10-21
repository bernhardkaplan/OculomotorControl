import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../PlottingScripts/")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from plot_bcpnn_traces import TracePlotter
import BCPNN
import nest
import numpy as np 
import pylab



def plot_voltages(volt_recorder):
    volt_data = nest.GetStatus(volt_recorder)[0]['events']
    gids = volt_data['senders']
    idx_0 = (gids == 1).nonzero()[0]
    idx_1 = (gids == 2).nonzero()[0]
    volt_0 = volt_data['V_m'][idx_0]
    volt_1 = volt_data['V_m'][idx_1]
    times_0 = volt_data['times'][idx_0]
    times_1 = volt_data['times'][idx_1]

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(times_0, volt_0, label='pre neuron')
    ax.plot(times_1, volt_1, label='post neuron')

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Volt [mV]')
    pylab.legend()


if __name__ == '__main__':

    t_sim = 400

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
    nest.SetStatus(voltmeter, [{"to_file": True, "withtime": True, 'label' : 'volt'}])
    nest.ConvergentConnect(voltmeter, nrns)
    exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label': 'spikes'})
    nest.ConvergentConnect(nrns, exc_spike_recorder)


    source_0 = nest.Create('spike_generator', 1)
    # pre synaptic spikes
#    spike_times_0 = np.array([50., 70., 100., 300., 350., 400.])
    spike_times_0 = np.array([])
    # post synaptic spikes 
    spike_times_1 = np.array([51., 71., 101., 201., 221., 231., 241., 300., 350., 400.])
    nest.SetStatus(source_0, {'spike_times' : spike_times_0})

    delta_t = 3.
    source_1 = nest.Create('spike_generator', 1)
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
    syn_params  = {'p_i': p_i, 'p_j': p_j, 'p_ij': p_ij, 'gain': 1.0, 'K': 1.0, \
            'fmax': fmax, 'epsilon': epsilon, 'delay':1.0, \
            'tau_i': tau_i, 'tau_j': tau_j, 'tau_e': tau_e, 'tau_p': tau_p}
    print 'syn_params init:', syn_params
    nest.Connect([nrns[0]], [nrns[1]], syn_params, model='bcpnn_synapse')


    nest.Simulate(t_sim)

    plot_voltages(voltmeter)

    spike_data = nest.GetStatus(exc_spike_recorder)[0]['events']
    gids = spike_data['senders']
    idx_0 = (gids == 1).nonzero()[0]
    st_0 = spike_data['times'][idx_0]

    idx_1 = (gids == 2).nonzero()[0]
    st_1 = spike_data['times'][idx_1]
    
    print 'spike_times pre:', st_0
    print 'spike_times post:', st_1

    #nest.SetStatus(nest.GetConnections([nrns[0]], [nrns[1]]), {'K': .7, 'gain': 0., 't_k': nest.GetKernelStatus()['time']})

    conns = nest.GetConnections([nrns[0]], [nrns[1]], synapse_model='bcpnn_synapse') # get the list of connections stored on the current MPI node
    if len(conns) > 0:
        cp = nest.GetStatus([conns[0]])  # retrieve the dictionary for this connection
        pi = cp[0]['p_i']
        pj = cp[0]['p_j']
        pij = cp[0]['p_ij']
        w = np.log(pij / (pi * pj))
        print 'weight after simulation:', w

    #######################
    # OFFLINE COMPUTATION
    #######################
    s_pre = BCPNN.convert_spiketrain_to_trace(st_0, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(st_1, t_sim)
    K_vec = syn_params['K'] * np.ones(s_pre.size)

    s_pre = BCPNN.convert_spiketrain_to_trace(s_pre, t_sim)
    s_post = BCPNN.convert_spiketrain_to_trace(s_post, t_sim)

#    K_vec[1500:2000] = .7
#    K_vec[2000:2500] = .3
#    K_vec[2500:] = .1
#    print 'K_vec:', K_vec

    bcpnn_traces = []
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, syn_params, K_vec=K_vec)
    bcpnn_traces = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]

    TP = TracePlotter({})

    dt = 0.1
    TP.plot_trace_with_spikes(bcpnn_traces, syn_params, dt, output_fn=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
            extra_txt=None)

#    TP.plot_trace(self, bcpnn_traces, bcpnn_params, dt, output_fn=None, info_txt=None, fig=None, \
#            color_pre='b', color_post='g', color_joint='r', style_joint='-')


    pylab.show() 
