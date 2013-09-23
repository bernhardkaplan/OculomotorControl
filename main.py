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
import time

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
    t0 = time.time()

    VI = VisualInput.VisualInput(params)
    MT = MotionPrediction.MotionPrediction(params, VI)
    pc_id, n_proc = MT.pc_id, MT.n_proc
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(params)
    CC = CreateConnections.CreateConnections(params)
#    CC.connect_mt_to_bg(MT, BG)

    next_state = [None, None, None, None] 
    actions = np.zeros((params['n_iterations'], 2))
    network_states_net_net = np.zeros((params['n_iterations'], 5))
    for iteration in xrange(params['n_iterations']):
        # integrate the real world trajectory and the eye direction and compute spike trains from that
        stim = VI.compute_input(MT.local_idx_exc, action_code=next_state)
        if params['debug_mpn']:
            save_spike_trains(params, iteration, stim)

        MT.update_input(stim) # run the network for some time 
        nest.Simulate(params['t_iteration'])
        state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
        network_states_net_net[iteration, :] = state_
        print 'Iteration: %d\tState before action: ' % (iteration), state_
        next_state = BG.select_action(state_) # BG returns the network_states_net of the next stimulus
        actions[iteration, :] = next_state
        print 'Iteration: %d\tState after action: ' % (iteration), next_state
#        VI.update_retina_image(BG.get_eye_direction())

    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['network_states_fn'], network_states_net_net)
    t1 = time.time() - t0
    print 'Time: %d [sec]' % t1

