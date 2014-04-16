import nest

def get_local_indices(pop):
    """
    Returns the GIDS assigned to the process.
    """
    local_nodes = []
    node_info   = nest.GetStatus(pop)
    for i_, d in enumerate(node_info):
        if d['local']:
            local_nodes.append(d['global_id'])
    return local_nodes
    
n_local_threads = 2
folder_name = 'Temp/'
nest.SetKernelStatus({'data_path':folder_name, 'overwrite_files': True, "local_num_threads":  n_local_threads})

n_cells = 10
n_pop = 4

populations = {}
populations_2 = {}
recorder = {}
local_gids = []

for i_ in xrange(n_pop):
    populations[i_] = nest.Create('iaf_cond_exp', n_cells)
    recorder[i_] = nest.Create('spike_detector', n_cells)

for i_ in xrange(n_pop):
    local_gids += get_local_indices(populations[i_])

n_local = len(local_gids)
stimulus = nest.Create('spike_generator', n_local)
#stimulus = nest.Create('spike_generator', n_cells * n_pop)

for i_ in xrange(n_pop):
    populations_2[i_] = nest.Create('iaf_cond_exp', n_cells)

nest_pc_id, nest_n_proc = nest.Rank(), nest.NumProcesses()
print 'PID: %d local_gids(pop1):' % (nest_pc_id), local_gids
print 'PID: %d populations:' % (nest_pc_id), populations
print 'PID: %d populations_2:' % (nest_pc_id), populations_2
