import os
import time
import utils

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



t0 = time.time()
#argv_1 = 'Training_DEBUG_RBL_titer25_2500_nStim6x1_taup50000_gain1.00/'
argv_1 = 'Training_RBL_mixed_titer25_2500_nStim3x8_taup50000_gain1.00/'
cmd = 'python PlottingScripts/plot_bcpnn_traces.py'


list_of_jobs = []


"""
    For a parameter sweep, give two addition parameters to the plotting script:
    1) the script count 
        --> to let the plotting script write in its own output file
    2) the parameter identifier
        which is used to speed up the analysis, see analyse_bcpnn_traces.py

"""
script_cnt = 0
for action_idx in [9, 10]:
    param_set_id = 0
#    for tau_p in [5000.]:
#        for tau_e in [1., 2., 5., 10., 50.]:
#            for tau_i in [1., 50., 100., 200.]:
    for tau_p in [5000., 10000., 50000.]:
        for tau_e in [1., 2., 5., 10., 50., 100., 200., 500., 5000., 15., 20., 30., 40., 75., 150.]:
            for tau_i in [1., 2., 5., 10., 50., 100., 200., 500., 15., 20., 30., 40., 75., 150.]:
                run_cmd = cmd + ' %s %d %d %d %d %d %d' % (argv_1, tau_i, tau_e, tau_p, action_idx, script_cnt, param_set_id)
                script_cnt += 1
                param_set_id += 1
                list_of_jobs.append(run_cmd)

# distribute the commands among processes
my_idx = utils.distribute_n(len(list_of_jobs), n_proc, pc_id) # this holds the indices for the jobs to be run by this processor
print 'pc_id %d job indices:' % pc_id, my_idx

for i_ in xrange(my_idx[0], my_idx[1]):
    job_name = list_of_jobs[i_]
    print 'pc_id %d runs:' % pc_id, job_name
    os.system(job_name)

t1 = time.time()
print 'Time pc_id %d: %d [sec] %.1f [min]' % (pc_id, t1 - t0, (t1 - t0)/60.)

#for thing in os.listdir('.'):

#    if thing.find('Test_afterRBL_2_it15__0-2') != -1:
#        run_cmd = cmd + ' ' + thing
#        print run_cmd
#        os.system(run_cmd)


