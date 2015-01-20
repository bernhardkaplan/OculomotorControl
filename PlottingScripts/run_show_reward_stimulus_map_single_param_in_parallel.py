import os, sys, inspect
# use this if you want to include modules from a subforder
#cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
#if cmd_subfolder not in sys.path:
#    sys.path.insert(0, cmd_subfolder)
import time
#import utils
import numpy as np
import subprocess

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
cmd = 'python'
script_name = 'PlottingScripts/show_reward_stimulus_map_single_param.py'
list_of_jobs = []

def distribute_n(n, n_proc, pid):
    """
    n: number of elements to be distributed
    pid: (int) process id of the process calling this function
    n_proc: total number of processors
    Returns the min and max index to be assigned to the processor with id pid
    """
    n_per_proc = int(n / n_proc)
    R = n % n_proc
    offset = min(pid, R)
    n_min = int(pid * n_per_proc + offset)
    if (pid < R):
        n_max = int(n_min + n_per_proc + 1)
    else:
        n_max = int(n_min + n_per_proc)
    return (n_min, n_max)


"""
    For a parameter sweep, give two addition parameters to the plotting script:
    1) the script count 
        --> to let the plotting script write in its own output file
    2) the parameter identifier
        which is used to speed up the analysis, see analyse_bcpnn_traces.py

    cnt_ = int(sys.argv[1])
    speed_mult_0 = float(sys.argv[2])
    speed_mult_1 = float(sys.argv[3])
    n_actions = int(sys.argv[4])
    k_ = float(sys.argv[5])
    rew_tol = float(sys.argv[6])

"""

script_cnt = 0
#for speed_mult in [[0.5, 1.], [0.5, 1.5], [0.5, 2.], [1., 1.5], [1., 2.]]:
#    speed_mult_0 = speed_mult[0]
#    speed_mult_1 = speed_mult[1]
#    for n_actions in np.arange(13, 23, 2):
#        for k_ in [20, 40, 100, 200]:      
#            for rew_tol in np.arange(0.01, 0.05, 0.01):
arg_strings = []
for speed_mult in [[0.5, 1.]]:
    speed_mult_0 = speed_mult[0]
    speed_mult_1 = speed_mult[1]
    for n_actions in np.arange(17, 19, 2):
        for k_ in [40]:
            for rew_tol in np.arange(0.01, 0.05, 0.01):
#                run_cmd = cmd + ' %d %.1f %.1f %d %d %.1f' % (script_cnt, speed_mult_0, speed_mult_1, n_actions, k_, rew_tol)
#                list_of_jobs.append(run_cmd)
#                arg_strings += ['%s %d %.1f %.1f %d %d %.1f' % (script_name, script_cnt, speed_mult_0, speed_mult_1, n_actions, k_, rew_tol))
                arg_strings.append([cmd, script_name, '%d' % script_cnt, '%.1f' % speed_mult_0, '%.1f' % speed_mult_1, '%d' % n_actions, '%d' % k_, '%.1f' % rew_tol])
                script_cnt += 1

# distribute the commands among processes
#my_idx = utils.distribute_n(len(list_of_jobs), n_proc, pc_id) # this holds the indices for the jobs to be run by this processor
#my_idx = utils.distribute_n(len(arg_strings), n_proc, pc_id) # this holds the indices for the jobs to be run by this processor
my_idx = distribute_n(len(arg_strings), n_proc, pc_id) # this holds the indices for the jobs to be run by this processor

for i_ in xrange(my_idx[0], my_idx[1]):
#    job_name = list_of_jobs[i_]
#    os.system(job_name)
#    subprocess.call(["ls", "-l"])
    print 'pc_id %d runs:' % pc_id, arg_strings[i_]
    subprocess.call(arg_strings[i_], shell=False)

t1 = time.time()
print 'Time pc_id %d: %d [sec] %.1f [min]' % (pc_id, t1 - t0, (t1 - t0)/60.)

#for thing in os.listdir('.'):

#    if thing.find('Test_afterRBL_2_it15__0-2') != -1:
#        run_cmd = cmd + ' ' + thing
#        print run_cmd
#        os.system(run_cmd)


