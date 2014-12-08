import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import utils
import simulation_parameters
import matplotlib
matplotlib.use('Agg')


class MetaAnalysisClass(object):
    def __init__(self, argv, verbose=False):
        stim_range = None
        if len(argv) == 1:
            if verbose:
                print 'Case 1', argv
            network_params = simulation_parameters.global_parameters()  
            params = network_params.params
            self.run_super_plot(params, stim_range)

        elif len(argv) == 2:                                # plot_ [FOLDER]
            if verbose:
                print 'Case 2', argv
            folder_name = argv[1]
            params = utils.load_params(folder_name)
            self.run_super_plot(params, stim_range)
        elif len(argv) == 3: 
            if verbose:
                print 'Case 3', argv
            if argv[1].isdigit() and argv[2].isdigit(): #  plot_ [STIM_1] [STIM_2]
                stim_range = (int(argv[1]), int(argv[2]))
                network_params = simulation_parameters.global_parameters()  
                params = network_params.params
                self.run_super_plot(params, stim_range)
            else:
                for fn in argv[1:]:                         # plot_ [FOLDER] [FOLDER]
                    params = utils.load_params(fn)
                    self.run_super_plot(params, stim_range)
        elif len(argv) == 4:                                
            if verbose:
                print 'Case 4', argv
            folder_name = argv[1]
            if argv[2].isdigit() and argv[3].isdigit(): # plot_ [FOLDER] [STIM_1] [STIM_2]
                stim_range = (int(argv[2]), int(argv[3]))
                params = utils.load_params(folder_name)
                # create one figure for the full stim range
                self.run_super_plot(params, stim_range)

                # create separate figures for each individual stimulus
#                for i_stim in xrange(stim_range[0], stim_range[1]):
#                    self.run_super_plot(params, (i_stim, i_stim + 1))
            else:
                # create separate figures for each individual folder
                for fn in argv[1:]:                         # plot_ [FOLDER_1] [FOLDER_2] [FOLDER_3]
                    params = utils.load_params(fn)
                    self.run_super_plot(params, stim_range)
        elif len(argv) > 4:                                 # plot_ [FOLDER_1] [FOLDER_2] .... [FOLDER_N]
            # create separate figures for each individual folder
            if verbose:
                print 'Case 5', argv
            for fn in argv[1:]:
                params = utils.load_params(fn)
                self.run_super_plot(params, stim_range)



    def run_super_plot(self, params, stim_range):
        """
        if stim_range == None:
            plot full range of iterations
        """
        raise NotImplementedError, 'To be implemented by sub class'
            


