import sys
import os
import numpy as np
import json
import time
import nest
import VisualInput
import MotionPrediction
import BasalGanglia
import simulation_parameters
import CreateConnections
import utils
from main_training import remove_files_from_folder, save_spike_trains
from PlottingScripts.PlotBGActivity import run_plot_bg
from PlottingScripts.PlotMPNActivity import MetaAnalysisClass

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


if __name__ == '__main__':

    t0 = time.time()

    assert (len(sys.argv) > 1), 'Missing training folder as command line argument'
    training_folder = os.path.abspath(sys.argv[1]) 
    print 'Training folder:', training_folder

    GP = simulation_parameters.global_parameters()
    if comm != None:
        comm.Barrier()

    write_params = True
    testing_params = GP.params
    if len(sys.argv) < 3:
        testing_params = GP.params
    else:
        testing_params_json = utils.load_params(os.path.abspath(sys.argv[2]))
        testing_params = utils.convert_to_NEST_conform_dict(testing_params_json)
        write_params = False
    

    if testing_params['training']:
        print 'Set training = False!'
        exit(1)

    testing_params['training_folder'] = training_folder
    if pc_id == 0 and write_params:
        GP.write_parameters_to_file(testing_params['params_fn_json'], testing_params) # write_parameters_to_file MUST be called before every simulation

    if comm != None:
        comm.Barrier()

#    training_params_fn = os.path.abspath(training_folder) + '/Parameters/simulation_parameters.json'
    training_params = utils.load_params(training_folder)
    actions = np.zeros((testing_params['n_iterations'] + 1, 3)) # the first row gives the initial action, [0, 0] (vx, vy, action_index)
    network_states_net= np.zeros((testing_params['n_iterations'], 4))
    training_stimuli = np.zeros((training_params['n_stim_training'], 4))
    training_stimuli_= np.loadtxt(training_params['training_sequence_fn'])
    training_stimuli = training_stimuli_
#    print 'debug', training_stimuli.shape
#    print 'debug', training_params['training_sequence_fn']
#    training_stimuli.reshape((training_params['n_stim_training'], 4))

    training_params['training_params'] = training_params # double check

    if pc_id == 0:
        remove_files_from_folder(testing_params['spiketimes_folder'])
        remove_files_from_folder(testing_params['input_folder_mpn'])
    
    t1 = time.time() - t0
    print 'Time1: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    VI = VisualInput.VisualInput(testing_params, comm=comm)
    t1 = time.time() - t0
    print 'Time2: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    MT = MotionPrediction.MotionPrediction(testing_params, VI, comm)
    t1 = time.time() - t0
    print 'Time3: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    BG = BasalGanglia.BasalGanglia(testing_params, comm)
    t1 = time.time() - t0
    print 'Time4: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    BG.write_cell_gids_to_file()

    CC = CreateConnections.CreateConnections(testing_params, comm)

    if comm != None:
        comm.Barrier()

    t1 = time.time() - t0
    print 'Time5: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

    CC.connect_mt_to_bg_after_training(MT, BG, training_params, testing_params, debug=True)
    if testing_params['connect_d1_after_training']:
        CC.connect_d1_after_training(BG, training_params, testing_params)
    t1 = time.time() - t0
    print 'Time6: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    if comm != None:
        comm.Barrier()
    BG.set_bias('d1')
    if comm != None:
        comm.Barrier()
    if testing_params['with_d2']:
        BG.set_bias('d2')
    if comm != None:
        comm.Barrier()
    t1 = time.time() - t0
    print 'Time7: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

    if len(testing_params['test_stim_range']) > 1:
        assert (testing_params['test_stim_range'][1] <= training_params['n_training_cycles'] * training_params['n_training_stim_per_cycle']), 'Corretct test_stim_range in sim params!'
    iteration_cnt = 0
    for i_, i_stim in enumerate(testing_params['test_stim_range']):
#        print 'DEBUG', training_stimuli, training_stimuli.shape, i_stim
        if len(training_stimuli.shape) == 1:
            VI.current_motion_params = training_stimuli
        else:
            VI.current_motion_params = training_stimuli[i_stim, :]
        for it in xrange(testing_params['n_iterations_per_stim']):

            if it >= (testing_params['n_iterations_per_stim'] - testing_params['n_silent_iterations']):
                stim, supervisor_state = VI.set_empty_input(MT.local_idx_exc)
            else:
                # integrate the real world trajectory and the eye direction and compute spike trains from that
                # and get the state information BEFORE MPN perceives anything
                # in order to set a supervisor signal
                stim, supervisor_state = VI.compute_input(MT.local_idx_exc, actions[iteration_cnt, :])

            if testing_params['debug_mpn']:
                print 'Iteration %d: Saving spike trains...' % iteration_cnt
                save_spike_trains(testing_params, iteration_cnt, stim, MT.local_idx_exc)
            MT.update_input(stim)
            if comm != None:
                comm.Barrier()
            nest.Simulate(testing_params['t_iteration'])
            if comm != None:
                comm.Barrier()

            state_ = MT.get_current_state(VI.tuning_prop_exc) # returns (x, y, v_x, v_y, orientation)

            network_states_net[iteration_cnt, :] = state_
            print 'Iteration: %d\t%d\tState before action: ' % (iteration_cnt, pc_id), state_

            next_action = BG.get_action() # BG returns the network_states_net of the next stimulus
            actions[iteration_cnt + 1, :] = next_action
            print 'Iteration: %d\t%d\tState after action: ' % (iteration_cnt, pc_id), next_action
            iteration_cnt += 1
            if comm != None:
                comm.Barrier()

    # DEEEEEEEEEBUG
#    CC.debug_mpn_connections(MT.exc_pop)
#    CC.get_weights(MT, BG, model='static_synapse')
#    debug_txt_d1 = CC.debug_connections(BG.strD1)
#    f = file('delme_conn_d1_%.1f_%d.txt' % (testing_params['gain_MT_d1'], pc_id), 'w')
#    f.write(debug_txt_d1)

#    CC.get_weights(MT, BG, model='static_synapse')
#    debug_txt_d2 = CC.debug_connections(BG.strD2)
#    f = file('delme_conn_d2_%.1f_%d.txt' % (testing_params['gain_MT_d1'], pc_id), 'w')
#    f.write(debug_txt_d2)

    t1 = time.time() - t0
    print 'Time8: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)
    if pc_id == 0:
        np.savetxt(testing_params['actions_taken_fn'], actions)
        np.savetxt(testing_params['network_states_fn'], network_states_net)
        np.savetxt(testing_params['motion_params_fn'], VI.motion_params)
        utils.remove_empty_files(testing_params['connections_folder'])
        utils.remove_empty_files(testing_params['spiketimes_folder'])
#        if params['n_stim'] > 6:
#            n_stim = 6 
#        else:
#            n_stim = params['n_stim']
        n_stim = testing_params['n_stim']
        run_plot_bg(testing_params, (0, n_stim))
        MAC = MetaAnalysisClass(['dummy', testing_params['folder_name'], str(0), str(n_stim)])
        MAC = MetaAnalysisClass([testing_params['folder_name']])
        run_plot_bg(testing_params, None)

    if comm != None:
        comm.Barrier()
    t1 = time.time() - t0
    print 'Time9: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

    t1 = time.time() - t0
    print 'TimeEND: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

