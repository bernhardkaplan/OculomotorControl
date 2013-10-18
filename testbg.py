import sys
import os
import VisualInput
import MotionPrediction
import BasalGanglia
import json
import simulation_parameters
import CreateConnections
import nest
import numpy as np
import BasalGangliaparameters
import pylab as pl

'''
export PYTHONPATH=/home/freeman/opt/nest/lib/python2.7/site-packages
export LD_LIBRARY_PATH=/lib/:/home/freeman/opt/nest/lib/nest/ 
lamboot
'''

nest.ResetKernel()
# load bcpnn synapse module and iaf neuron with bias
if (not 'bcpnn_synapse' in nest.Models('synapses')):
    nest.Install('pt_module')
nest.SetKernelStatus({"overwrite_files": True})

       

def save_spike_trains(params, iteration, stim_list):
    n_units = len(stim_list)
    fn_base = params['input_st_fn_mpn']
    for i_ in xrange(n_units):
        if len(stim_list[i_]) > 0:
            fn = fn_base + '%d_%d.dat' % (iteration, i_)
            np.savetxt(fn, stim_list[i_])
	
if __name__ == '__main__':
    if len(sys.argv) > 1: # re-run an old parameter file
   	param_fn = sys.argv[1]
        if os.path.isdir(param_fn): # go to the path containing the json object storing old parameters
            param_fn += '/Parameters/simulation_parameters.json' # hard coded subpath in ParameterContainer
        assert os.path.exists(param_fn), 'ERROR: Can not find %s - please give an existing parameter filename or folder name to re-run a simulation' % (param_fn)
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
        

    else: # run a simulation with parameters as set in simulation_parameters
        GP = simulation_parameters.global_parameters()
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
        params = GP.params
	pBG = BasalGangliaparameters.BasalGangliaParameters()
#	pBG.write_parameters_to_file()
	paramsBG = pBG.params

    BG = BasalGanglia.BasalGanglia(paramsBG)
    for iterations in range(params['n_iterations']):
	    state = iterations % paramsBG['n_states']
    	    BG.set_state(state)
   	    nest.Simulate(params['t_iteration'])
    	    action = BG.get_action()
	    print BG.recorder_output 
    pl.figure(51)
#all = len(nest.GetStatus(voltmeter_in2)[0]['events']['V_m'])
    pl.rc("font", size=newfontsize)
    z = 0
    color = ['b','g', 'r', 'c', 'm', 'y', 'k']
    for a in [BG.recorder_output]:
    	cl = color[z%len(color)]
    	pl.scatter(nest.GetStatus(a)[0]['events']['times'], nest.GetStatus(a)[0]['events']['senders'], c=cl, marker='.')
    	z+=1
    pl.axis([ 0, nest.GetStatus(voltmeter_in1)[0]['n_events'], 0, 300 ])
    frame51 = pl.gca()
    frame51.xaxis.label.set_fontsize(labels_fontsize)
    frame51.yaxis.label.set_fontsize(labels_fontsize)
    pl.xlabel("time "+ r"$ms$")
    pl.show()
"""
   VI = VisualInput.VisualInput(params) # pass parameters to the VisualInput module
    MT = MotionPrediction.MotionPrediction(params)
    BG = BasalGanglia.BasalGanglia(params)
    CC = CreateConnections.CreateConnections(params)
    CC.connect_mt_to_bg(MT, BG)

    for iteration in xrange(params['n_iterations']):
        # integrate the real world trajectory and the eye direction and compute spike trains from that
        state_old = iteration % params['n_states'] 
        stim = VI.compute_input(params['t_iteration'], MT.local_idx_exc, stim_state=state_old)
        if params['debug_mpn']:
            save_spike_trains(params, iteration, stim)
        MT.update_input(stim) # run the network for some time 
        nest.Simulate(params['t_iteration'])
        state_new = MT.get_current_state()
        print 'Iteration: %d\tState before action: %d' % (iteration, state_new)
        state_old = BG.select_action(state_new) # BG returns vx_eye
        VI.update_retina_image(BG.get_eye_direction())
"""
    
