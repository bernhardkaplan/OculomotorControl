import simulation_parameters
import VisualInput
import MotionPrediction
import BasalGanglia


class WorldTimer(object):

    def __init__(self):
        """
        Initialize timer, parameters
        """
        self.time_step = 0 # [ms]


    def advance(self):
        """
        Advance time step counter variable
        """
        self.time_step += 1 # [ms]



if __name__ == '__main__':

    GP = simulation_parameters.global_parameters()
    params = GP.params
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

    GP.write_parameters_to_file()

    VI = VisualInput.VisualInput(params)
    MT = MotionPrediction(params)
    BG = BasalGanglia(params)

    for iteration in xrange(params['n_iterations']):
        stim = VI.compute_input(params['t_integrate'])
        MT.compute_xpred(stim)

#    WT = WorldTimer()

#    """
#    First the stimulus is computed for a certain time
#    """
#    stimulus = VI.compute_input(params['t_integrate'])


#    """
#    Based on the stimulus position (in visual field (retinotopic) coordinates)
#    the MotionPrediction class computes an estimate of the stimulus position 'xpred'
#    """
#    xpred = MT.compute_xpred(stimulus)


#    """
#    The predicted xposition is passed on to the BasalGanglia which takes a decision and selects an action.
#    The action is the eye movement (in which direction and how strong the eye should be moved).
#    """
#    action = BG.select_action(xpred) 


#    """
#    ... Omited steps: Superior collicus transforms BG signals originating from the Pars reticulata of the Substantia nigra (a part in BG) 
#    into motor signals for the Oculomotor nerve
#    Based on the output action the new stimulus position in retinotopic coordinates are computed which determines the next stimulus.
#    """
#    VI.transform_image(action) # linear transformation 

