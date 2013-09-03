import sys
import os
import VisualInput
import MotionPrediction
import BasalGanglia
import VisualInputParameters
import json
import simulation_parameters
import CreateConnections


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

    # TODO: update VisualInputParameters and possibly remove it completely
    # decide whether to load the default parameters or to take parameters from a file
    VIP = VisualInputParameters.VisualInputParameters()
    VIP.update_values(params)
    vi_params = VIP.params

    VI = VisualInput.VisualInput(vi_params) # pass parameters to the VisualInput module

    MT = MotionPrediction.MotionPrediction(params)
    BG = BasalGanglia.BasalGanglia(params)
    CC = CreateConnections.CreateConnections(params)
    CC.connect_mt_to_bg(MT, BG)


    for iteration in xrange(params['n_iterations']):
        # integrate the real world trajectory and the eye direction and compute spike trains from that
        stim = VI.compute_input(params['t_iteration'], BG.get_eye_direction())
        MT.compute_state(stim) # run the network for some time 
        BG.move_eye(MT.current_state)


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
