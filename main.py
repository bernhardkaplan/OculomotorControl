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
import os
import pylab as pl
try: 
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"



def save_spike_trains(params, iteration, stim_list, gid_list):
    n_units = len(stim_list)
    fn_base = params['input_st_fn_mpn']
    for i_ in xrange(n_units):
        if len(stim_list[i_]) > 0:
            fn = fn_base + '%d_%d.dat' % (iteration, gid_list[i_] - 1)
            np.savetxt(fn, stim_list[i_])


def remove_files_from_folder(folder):
    print 'Removing all files from folder:', folder
    path =  os.path.abspath(folder)
    cmd = 'rm  %s/*' % path
    print cmd
    os.system(cmd)


if __name__ == '__main__':

    t1 = time.time()
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

    weights_sim = {}
    staten2D1 = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
    staten2D2 = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]

    VI = VisualInput.VisualInput(params)
    MT = MotionPrediction.MotionPrediction(params, VI, comm)

    if pc_id == 0:
        remove_files_from_folder(params['spiketimes_folder_mpn'])
        remove_files_from_folder(params['input_folder_mpn'])
    
    VI.set_pc_id(pc_id)
    BG = BasalGanglia.BasalGanglia(params, comm)
    CC = CreateConnections.CreateConnections(params, comm)
    CC.connect_mt_to_bg(MT, BG)

    actions = np.zeros((params['n_iterations'] + 1, 2)) # the first row gives the initial action, [0, 0] (vx, vy)
    network_states_net= np.zeros((params['n_iterations'], 4))
    for iteration in xrange(params['n_iterations']):

        # integrate the real world trajectory and the eye direction and compute spike trains from that
        # and get the state information BEFORE MPN perceives anything
        # in order to set a supervisor signal
        stim, supervisor_state = VI.compute_input(MT.local_idx_exc, action_code=actions[iteration, :])

        print 'DEBUG iteration %d pc_id %d current motion params: (x,y) (u, v)' % (iteration, pc_id), VI.current_motion_params[0], VI.current_motion_params[1], VI.current_motion_params[2], VI.current_motion_params[3]
        print 'Iteration: %d\t%d\tsupervisor_state : ' % (iteration, pc_id), supervisor_state
        BG.supervised_training(supervisor_state)

        if params['debug_mpn']:
            print 'Saving spike trains...'
            save_spike_trains(params, iteration, stim, MT.local_idx_exc)

        # compute BG input (for supervised learning)
#        target_action = VI.transform_trajectory_to_action()
        # BG.update_input(stim) #--> updates the Poisson-populations coding for the state
        # BG.train_action_output(target_action)

        # remove MT.update_input etc
        MT.update_input(stim) # run the network for some time 
        if comm != None:
            comm.barrier()
        nest.Simulate(params['t_iteration'])
        if comm != None:
            comm.barrier()

        state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)

        for nactions in range(params['n_actions']):
            conn = nest.GetConnections(source = MT.exc_pop, target = BG.strD1[nactions], synapse_model = 'bcpnn_synapse')
            staten2D1[iteration][nactions] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]) #BG.params['params_synapse_d1_MT_BG']['gain'] * 
            conn = nest.GetConnections(source = MT.exc_pop, target = BG.strD2[nactions], synapse_model = 'bcpnn_synapse')
            staten2D2[iteration][nactions] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
#        BG.update_poisson_layer(state_)
        weights_sim[iteration] = BG.get_weights(MT.exc_pop, BG.strD1[0])
        network_states_net[iteration, :] = state_
        print 'Iteration: %d\t%d\tState before action: ' % (iteration, pc_id), state_
        next_state = BG.get_action(state_) # BG returns the network_states_net of the next stimulus
        actions[iteration + 1, :] = next_state
        print 'Iteration: %d\t%d\tState after action: ' % (iteration, pc_id), next_state
#        exit(1)
#        VI.update_retina_image(BG.get_eye_direction())

    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['network_states_fn'], network_states_net)
        np.savetxt(params['motion_params_fn'], VI.motion_params)


    t1 = time.time() - t0
    print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    BG.stop_supervisor()
    print 'supervised learning completed'
    CC.get_weights(MT, BG)

#   for test in range(params['n_iterations']/2, params['n_iterations']):
#       stim, supervisor_state = VI.compute_input(MT.local_idx_exc, action_code=actions[test, :])
#
#       print 'DEBUG iteration %d pc_id %d current motion params: (x,y) (u, v)' % (test, pc_id), VI.current_motion_params[0], VI.current_motion_params[1], VI.current_motion_params[2], VI.current_motion_params[3]
#
#
#
#       MT.update_input(stim) # run the network for some time 
#       if comm != None:
#           comm.barrier()
#       nest.Simulate(params['t_iteration'])
#       if comm != None:
#           comm.barrier()
#
#       state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)
#
#       for nactions in range(params['n_actions']):
#           conn = nest.GetConnections(source = MT.exc_pop, target = BG.strD1[nactions], synapse_model = 'bcpnn_synapse')
#           staten2D1[test][nactions] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]) #BG.params['params_synapse_d1_MT_BG']['gain'] * 
#           conn = nest.GetConnections(source = MT.exc_pop, target = BG.strD2[nactions], synapse_model = 'bcpnn_synapse')
#           staten2D2[test][nactions] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
#        BG.update_poisson_layer(state_)
#       weights_sim[test] = BG.get_weights(MT.exc_pop, BG.strD1[0])
#       network_states_net[test, :] = state_
#       print 'Iteration: %d\t%d\tState before action: ' % (test, pc_id), state_
#       next_state = BG.get_action(state_) # BG returns the network_states_net of the next stimulus
#       actions[test + 1, :] = next_state
#       print 'Iteration: %d\t%d\tState after action: ' % (test, pc_id), next_state
#       test += 1
#   t1 = time.time() - t0
#   print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)


#    print 'weight simu ', weights_sim
    pl.figure(1)
    pl.subplot(211)
    pl.plot(staten2D1)
    pl.ylabel(r'$w_{0j}$')
    pl.subplot(212)
    pl.plot(staten2D2)
    pl.ylabel(r'$w_{0j}$')
    pl.xlabel('trials')
    pl.show()
