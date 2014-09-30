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
argv_1 = 'Training_RBL_3_titer25__3_nStim2x3_taup50000_gain0.05'
cmd = 'python PlottingScripts/tune_bcpnn_trace_params.py'

list_of_jobs = []

#for tau_zi in [5., 10., 20., 50., 100., 200., 400.]:
#    for tau_zj in [5., 10., 20., 50., 100., 200., 400.]:
#        for tau_e in [5., 50., 500., 5000., 50000., 250000.]:
#            for tau_p in [5., 50., 500., 5000., 50000., 250000.]:

for tau_zi in [5., 10., 20., 50., 100., 200.]:
    for tau_zj in [5., 10., 20., 50., 100., 200.]:
        for tau_e in [5., 50., 100., 500., 5000., 50000.]:
            for tau_p in [10000., 50000.]:
                run_cmd = cmd + ' %s %d %d %d %d' % (argv_1, tau_zi, tau_zj, tau_e, tau_p)
    #            print run_cmd
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


