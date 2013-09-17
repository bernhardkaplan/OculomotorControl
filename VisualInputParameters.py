import os
import numpy as np
import json
import ParameterContainer

class VisualInputParameters(ParameterContainer.ParameterContainer):
    def __init__(self, params_fn=None):
        super(VisualInputParameters, self).__init__(params_fn)
        if params_fn == None:
            self.set_default_params()


    def set_default_params(self):

        self.set_visual_input_params()



    def set_filenames(self, main_folder=None):
        if main_folder == None:
            main_folder = 'AwesomeResults/'
        super(VisualInputParameters, self).set_filenames(main_folder)
#        print 'Folder names:', self.params['folder_names']
