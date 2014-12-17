import sys
import os
import simulation_parameters
import utils
import numpy as np
import json
import pylab
import BasalGanglia

if __name__ == '__main__':

    params = utils.load_params(sys.argv[1])

    problematic_stimuli_idx = np.loadtxt(params['data_folder'] + 'problematic_stimuli_idx.txt')

    for i_, idx in enumerate(problematic_stimuli_idx):
        cmd = 'python PlottingScripts/PlotBGActivity.py %s %d %d' % (params['folder_name'], idx, idx + 1)
        print 'run:', cmd
        os.system(cmd)
