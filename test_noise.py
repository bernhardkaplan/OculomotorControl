import nest
import numpy as np
import sys
import simulation_parameters
import utils
import pylab
import subprocess


def test_noise(params, t_sim=None):
    if t_sim == None:
        t_sim = 10000. + params['dt']
    n_nrns = 20

    
    nest.SetKernelStatus({'data_path': 'TestNoise/', 'overwrite_files': True, 'resolution' : params['dt']})

    cell_params = params['cell_params_exc_mpn'].copy()
    nrns = nest.Create(params['neuron_model_mpn'], n_nrns, params=cell_params)

#    nest.Install('pt_module')
#    cell_params = params['param_msn_d1'].copy()
#    nrns = nest.Create(params['model_exc_neuron'], n_nrns, params=cell_params)

    noise_exc = nest.Create('poisson_generator', n_nrns)
    noise_inh = nest.Create('poisson_generator', n_nrns)
    nest.SetStatus(noise_exc, {'rate' : params['f_noise_exc']})
    nest.SetStatus(noise_inh, {'rate' : params['f_noise_inh']})

    nest.Connect(noise_exc, nrns, params['w_noise_exc'], params['dt'])

    voltmeter = nest.Create('multimeter', params={'record_from': ['V_m']})
    nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'test_noise', 'interval': params['dt_volt']}])
    nest.DivergentConnect(voltmeter, nrns)

    nest.Simulate(t_sim) 

    fns = utils.find_files('TestNoise/', 'test_noise')
    print 'fns', fns

    volt_mean_std = np.zeros((n_nrns, 2))
    all_traces = np.zeros((t_sim / params['dt_volt'] - 1, n_nrns))
    mean_trace = np.zeros((t_sim / params['dt_volt'] - 1, 2))
    for fn in fns:
        path = 'TestNoise/' + fn
        reply = subprocess.check_output(['wc', '-l', '%s' % path])
        print 'reply', reply
        d = np.loadtxt(path)
        gids = np.unique(d[:, 0])
        for i_, gid in enumerate(gids):
            time_axis, volt = utils.extract_trace(d, gid)
            volt_mean_std[i_, 0] = volt.mean()
            volt_mean_std[i_, 1] = volt.std()
            all_traces[:, i_] = volt
    
    for t_ in xrange(int(t_sim / params['dt_volt'] - 1)):
        mean_trace[t_, 0] = all_traces[t_, :].mean()
        mean_trace[t_, 1] = all_traces[t_, :].std()

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for i_, gid in enumerate(gids):
        ax.plot(time_axis, all_traces[:, i_])

    ax.errorbar(time_axis, mean_trace[:, 0], yerr=mean_trace[:, 1], lw=3)
    print 'Average mean and fluctuation: %.2f +- %.2f' % (mean_trace[:, 0].mean(), mean_trace[:, 1].mean())



if __name__ == '__main__':
    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    test_noise(params, t_sim=1000.)

    pylab.show()
