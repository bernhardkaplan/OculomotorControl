import os
import numpy
import sys
import simulation_parameters

class MergeSpikefiles(object):

    def __init__(self, params):
        self.params = params


    def merge_nspike_files(self, merge_pattern, sorted_pattern_output,  sort_idx=1):
        rnd_nr1 = numpy.random.randint(0,10**8)
        rnd_nr2 = rnd_nr1 + 1
        fn_out = sorted_pattern_output
        print 'output_file:', fn_out
        # merge files from different processors
        tmp_file = "tmp_%d" % (rnd_nr2)
        os.system("cat %s_* > %s" % (merge_pattern,  tmp_file))
        # sort according to cell id
        os.system("sort -gk %d %s > %s" % (sort_idx, tmp_file, fn_out))
        os.system("rm %s" % (tmp_file))


    def merge_spiketimes_files(self, merge_pattern, sorted_pattern_output,  sort_idx=1):
        rnd_nr1 = numpy.random.randint(0,10**8)
        rnd_nr2 = numpy.random.randint(0,10**8) + 1
        fn_out = sorted_pattern_output
        print 'output_file:', fn_out
        # merge files from different processors
        tmp_file = "tmp_%d" % (rnd_nr2)
        print 'debug mergepattern', merge_pattern
        os.system("cat %s* > %s" % (merge_pattern,  tmp_file))
        # sort according to cell id
        os.system("sort -gk %d %s > %s" % (sort_idx, tmp_file, fn_out))
        os.system("rm %s" % (tmp_file))




if __name__ == '__main__':
    info_txt = \
    """
    Usage:
        python MergeSpikeFiles.py [FOLDER] [CELLTYPE] 
        or
        python MergeSpikeFiles.py [FOLDER] [CELLTYPE] [PATTERN_NUMBER]

    """
#    assert (len(sys.argv) > 2), 'ERROR: folder and cell_type not given\n' + info_txt
    try:
        folder = sys.argv[1]
        params_fn = os.path.abspath(folder) + '/Parameters/simulation_parameters.json'
        param_tool = simulation_parameters.global_parameters(params_fn=params_fn)
    except:
        param_tool = simulation_parameters.global_parameters()

    params = param_tool.params

    try:
        cell_type = sys.argv[2]
    except:
        cell_types = ['mpn_exc', 'mpn_inh']


    MS = MergeSpikefiles(params)
    for cell_type in cell_types:
#        print 'Merging nspike file for %s ' % (cell_type)
#        MS.merge_nspike_files(params['%s_spike_fn_base' % cell_type], params['%s_spikes_merged_fn_base' % cell_type])
        print 'Merging spiketimes file for %s ' % (cell_type)
        merge_pattern = params['spiketimes_folder_mpn'] + params['%s_spikes_fn' % cell_type]
        output_fn = params['spiketimes_folder_mpn'] + params['%s_spikes_fn_merged' % cell_type]
        MS.merge_spiketimes_files(merge_pattern, output_fn)
        
