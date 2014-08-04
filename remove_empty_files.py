import os
import sys
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


if len(sys.argv) > 1:
    utils.remove_empty_files(sys.argv[1])

else:
    list_of_folders = []
    for thing in os.listdir('.'):
        if thing.find('Test_') != -1:
            path = thing + '/' + 'Spikes/'
            list_of_folders.append(path)

    if USE_MPI:
        my_folder_idx = utils.distribute_n(len(list_of_folders), n_proc, pc_id)
        for i_ in xrange(my_folder_idx[0], my_folder_idx[1]):
            print 'Proc %d removes files from folder: %s' % (pc_id, list_of_folders[i_])
            utils.remove_empty_files(list_of_folders[i_])
    else:
        for path in list_of_folders:
            print 'Removing files from:', path
            utils.remove_empty_files(path)
