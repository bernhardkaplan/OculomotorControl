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
import utils
from copy import deepcopy
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



def remap_actions(params, delta_idx):
    remapped_actions = np.zeros(params['n_actions'], dtype=np.int)
    
    RNG = np.random
    RNG.seed(0)
    for i_ in xrange(0, params['n_actions']):
        a = i_
        while (a == i_):
            rnd_action = np.int(i_ + delta_idx * utils.get_plus_minus(RNG))
            a = min(max(0, rnd_action), params['n_actions'] - 1)
            remapped_actions[i_] = a
    return remapped_actions


if __name__ == '__main__':
    if len(sys.argv) > 1: # re-run an old parameter file
        params = utils.load_params(sys.argv[1])
    else:
        GP = simulation_parameters.global_parameters()
        params = GP.params
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation

    initial_action_mapping = remap_actions(params, 3)
    for i_ in xrange(initial_action_mapping.size):
        print i_, initial_action_mapping[i_]
    # use this action mapping to wire up each cell

    VI = VisualInput.VisualInput(params, comm=comm)
    training_stimuli = VI.create_training_sequence()
    np.savetxt(params['motion_params_fn'], VI.motion_params)
    MPN = MotionPrediction.MotionPrediction(params, VI, comm=comm)
    BG = BasalGanglia.BasalGanglia(params, comm, dummy=False)

    bg_gids = BG.write_cell_gids_to_file()

    n_cells = params['n_exc_mpn']
#    n_cells = 10
    M_init_d1 = np.zeros((n_cells, params['n_actions'] * params['n_cells_per_d1']))
    M_init_d2 = np.zeros((n_cells, params['n_actions'] * params['n_cells_per_d2']))
    gid_offset_d1 = min(bg_gids['d1'][0])
    gid_offset_d2 = min(bg_gids['d2'][0])

    np.random.seed(params['tuning_prop_seed'])
    # for each cell decide to which action_idx (i.e. the D1 population) it projects to

    conn_list_d1 = ''
    conn_list_d2 = ''

    std_weight = 0.0
    delay = 1.0
    for i_ in xrange(n_cells):
        optimal_action = BG.get_optimal_action_for_stimulus(VI.tuning_prop_exc[i_, :])
#        tgt_action_idx = initial_action_mapping[optimal_action[2]]
        tgt_action_idx = np.random.randint(0, params['n_actions'])
#        print 'connection cell %d to tgt_idx:' % (i_), tgt_action_idx, optimal_action
        for i_action in xrange(params['n_actions']):
            if i_action == tgt_action_idx:
                tgt_gids = bg_gids['d1'][tgt_action_idx]
                M_init_d1[i_, np.array(tgt_gids)- gid_offset_d1] = 1.0
                for tgt_gid in tgt_gids:
                    conn_list_d1 += '%d\t%d\t%.4e\t%1.e\n' % (i_ + 1, tgt_gid, std_weight, delay)
            else:
                tgt_gids = bg_gids['d2'][i_action]
                M_init_d2[i_, np.array(tgt_gids) - gid_offset_d2] = 1.0
                for tgt_gid in tgt_gids:
                    conn_list_d2 += '%d\t%d\t%.4e\t%.1e\n' % (i_ + 1, tgt_gid, std_weight, delay)

    output_fn = params['initial_weight_matrix_d1']
    print 'Saving to:', output_fn
    np.savetxt(output_fn, M_init_d1)
    output_fn = params['initial_weight_matrix_d2']
    print 'Saving to:', output_fn
    np.savetxt(output_fn, M_init_d1)
    np.savetxt(output_fn, M_init_d2)
    print 'Saving to:', params['mpn_bgd1_merged_conn_fn']
    fn_out = file(params['mpn_bgd1_merged_conn_fn'], 'w')
    fn_out.write(conn_list_d1)
    fn_out.flush()
    fn_out.close()
    print 'Saving to:', params['mpn_bgd2_merged_conn_fn']
    fn_out = file(params['mpn_bgd2_merged_conn_fn'], 'w')
    fn_out.write(conn_list_d2)
    fn_out.flush()
    fn_out.close()
