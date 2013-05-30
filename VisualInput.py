import numpy as np
import json

class VisualInput(object):

    def __init__(self, params):
        """
        Keyword arguments
        params -- dictionary that contains 
        """
        self.params = params
        self.trajectories = []

    def compute_input(self, t_integrate):
        trajectory = self.compute_stimulus_trajectory(t_integrate)
        self.compute_detector_response(trajectory)


    def compute_stimulus_trajectory(self, t_integrate):
        time_axis = np.arange(0, t_integrate, self.params['dt_stim'])
        trajectory = self.params['v_stim'] * time_axis + np.ones(t_integrate) * self.params['x_offset']
        self.trajectories.append(trajectory) # store for later save 
        return trajectory

    

    def compute_detector_response(self, trajectory):

        network_response = np.zeros((self.n_units, self.t_axis.size))

        for unit in xrange(self.n_units):
            network_response[unit, :] = np.exp(-.5 * ((trajectory - self.tuning_prop[unit, 0]) / blur_X)**2 \
                    - .5 * ((v_stim - self.tuning_prop[unit, 1]) / blur_V)**2)
