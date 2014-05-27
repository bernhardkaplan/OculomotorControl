import os
import numpy as np
import json
import subprocess


def run_analysis(training_folder, test_string):
    for fn in os.listdir('.'):
        if fn.find(test_string) != -1:
            if not os.path.exists(fn + '/Data/x_displacement_mean.dat'):
                print 'Found:', fn
    #            cmd = 'python PlottingScripts/compare_test_and_training_performance.py %s %s' % (training_folder, fn)
                cmd = ['python', 'PlottingScripts/compare_test_and_training_performance.py', training_folder, fn]
                try:
    #                os.system(cmd)
                    subprocess.check_call(cmd)
                except:
                    exit(1)
                    pass


def get_minimal_displacement(test_str):

    x_displ = []
    folder_names = {}
    sim_id = 0
    for i_, fn in enumerate(os.listdir('.')):
        if fn.find(test_str) != -1:
            path = fn + '/' + 'Data/x_displacement_mean.dat'
            f = file(path, 'r')
            d = json.load(f)
            x_displ.append(d['xdisplacement_mean'])
            folder_names[sim_id] = fn
            sim_id += 1

    all_displ = np.array(x_displ)
    min_idx = np.argmin(all_displ)
    min_displacement = all_displ[min_idx]
    print 'Minimal displacement:', min_displacement, 'from', folder_names[min_idx], 'average', all_displ.mean()
    


if __name__ == '__main__':
    test_str = 'Test_nRF500_d1recTrue_1-1_nStim5x20'
    training_folder = 'Training_nRF500_clipWeights1-1_nStim5x20_it25_tsim20000_taup10000/'

#    training_folder = 'Training_nRF500_clipWeights1-1_nStim5x50_it25_tsim50000_taup25000/'
#    test_str 'Test_nRF500_d1recTrue_1-1_nStim5x50'
    run_analysis(training_folder, test_str)
    get_minimal_displacement(test_str)
