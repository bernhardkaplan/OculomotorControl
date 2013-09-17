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
        self.t_axis = np.arange(0, self.params['t_iteration'], self.params['dt'])
        self.t_current = 0 # stores the 'current' time
        np.random.seed(self.params['visual_stim_seed'])


    def update_retina_image(self, eye_direction):
        pass

    def compute_input(self, t_integrate, local_gids, stim_state):
        """
        Integrate the real world trajectory and the eye direction and compute spike trains from that.

        Keyword arguments:
        t_integrate -- time for which the input is computed
        stim_state -- the index of the population which will be stimulated
        """
        return self.create_dummy_stim(t_integrate, local_gids, stim_state)
#        self.tuning_prop_exc = self.set_tuning_prop('exc')
#        self.tuning_prop_inh = self.set_tuning_prop('inh')
#        trajectory = self.compute_stimulus_trajectory(t_integrate)
#        self.compute_detector_response(trajectory)





    def create_dummy_stim(self, t_integrate, local_gids, stim_state=0):
        """
        Keyword arguments:
        t_integrate -- (float) length of the stimulus to be created
        local_gids -- list of gids for which a stimulus shall be created
        """
        print 'Creating dummy spike trains', self.t_current
#        stim = [ [] for unit in xrange(self.params['n_exc_per_mc'])]
        stim = [ [] for gid in xrange(len(local_gids))]

        for i_, gid in enumerate(local_gids):
            # get the cell from the list of populations
            mc_idx = (gid - 1) / self.params['n_exc_per_mc']
            idx_in_pop = (gid - 1) - mc_idx * self.params['n_exc_per_mc']
            if mc_idx == stim_state:
                n_spikes = np.random.randint(20, 50)
                stim[i_] = np.around(np.random.rand(n_spikes) * t_integrate + self.t_current, decimals=1)
                stim[i_] = np.sort(stim[i_])
        self.t_current += t_integrate
        return stim



    def compute_stimulus_trajectory(self, t_integrate):
        v_stim = self.params['motion_params'][2]
        trajectory = self.params['motion_params'][2] * time_axis + np.ones(t_integrate) * self.params['x_offset']
        self.trajectories.append(trajectory) # store for later save 
        return trajectory

    

    def compute_detector_response(self, trajectory):

        detector_response = np.zeros((self.params['n_exc'], self.t_axis.size))
        v_stim = self.params['motion_params'][2]

        for unit in xrange(self.params['n_exc']):
            print 'debug', trajectory.shape, detector_response[unit, :].shape
            detector_response[unit, :] = np.exp(-.5 * ((trajectory - self.tuning_prop_exc[unit, 0]) / self.params['blur_X'])**2 \
                    - .5 * ((v_stim - self.tuning_prop_exc[unit, 1]) / self.params['blur_V'])**2)


    def set_tuning_prop(self, cell_type, mode='hexgrid'):

        if self.params['n_grid_dimensions'] == 2:
            return self.set_tuning_prop_2D(mode, cell_type)
        else:
            return self.set_tuning_prop_1D(cell_type)


    def set_tuning_prop_1D(self, cell_type='exc'):

        np.random.seed(self.params['tuning_prop_seed'])
        if cell_type == 'exc':
            n_cells = self.params['n_exc']
            n_v = self.params['n_v']
            n_rf_x = self.params['n_rf_x']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        else:
            n_cells = self.params['n_inh']
            n_v = self.params['n_v_inh']
            n_rf_x = self.params['n_rf_x_inh']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        tuning_prop = np.zeros((n_cells, 5))
        if self.params['log_scale']==1:
            v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
        else:
            v_rho = np.logspace(np.log(v_min)/np.log(self.params['log_scale']),
                            np.log(v_max)/np.log(self.params['log_scale']), num=n_v,
                            endpoint=True, base=self.params['log_scale'])
        n_orientation = self.params['n_orientation']
        orientations = np.linspace(0, np.pi, n_orientation, endpoint=False)
        xlim = (0, self.params['torus_width'])

        RF = np.linspace(0, self.params['torus_width'], n_rf_x, endpoint=False)
        index = 0
        random_rotation_for_orientation = np.pi*np.random.rand(n_rf_x * n_v * n_orientation) * self.params['sigma_rf_orientation']

            # todo do the same for v_rho?
        for i_RF in xrange(n_rf_x):
            for i_v_rho, rho in enumerate(v_rho):
                for orientation in orientations:
                # for plotting this looks nicer, and due to the torus property it doesn't make a difference
                    tuning_prop[index, 0] = (RF[i_RF] + self.params['sigma_rf_pos'] * rnd.randn()) % self.params['torus_width']
                    tuning_prop[index, 1] = 0.5 # i_RF / float(n_rf_x) # y-pos 
                    tuning_prop[index, 2] = rho * (1. + self.params['sigma_rf_speed'] * rnd.randn())
                    tuning_prop[index, 3] = 0. # np.sin(theta + random_rotation[index]) * rho * (1. + self.params['sigma_rf_speed'] * rnd.randn())
                    tuning_prop[index, 4] = (orientation + random_rotation_for_orientation[index]) % np.pi

                    index += 1

        return tuning_prop




    def set_tuning_prop_2D(self, mode='hexgrid', cell_type='exc'):
        """
        Place n_exc excitatory cells in a 4-dimensional space by some mode (random, hexgrid, ...).
        The position of each cell represents its excitability to a given a 4-dim stimulus.
        The radius of their receptive field is assumed to be constant (TODO: one coud think that it would depend on the density of neurons?)

        return value:
            tp = set_tuning_prop(self.params)
            tp[:, 0] : x-position
            tp[:, 1] : y-position
            tp[:, 2] : u-position (speed in x-direction)
            tp[:, 3] : v-position (speed in y-direction)

        All x-y values are in range [0..1]. Positios are defined on a torus and a dot moving to a border reappears on the other side (as in Pac-Man)
        By convention, velocity is such that V=(1,0) corresponds to one horizontal spatial period in one temporal period.
        This implies that in one frame, a translation is of  ``1. / N_frame`` in cortical space.
        """

        rnd.seed(self.params['tuning_prop_seed'])
        if cell_type == 'exc':
            n_cells = self.params['n_exc']
            n_theta = self.params['n_theta']
            n_v = self.params['n_v']
            n_rf_x = self.params['n_rf_x']
            n_rf_y = self.params['n_rf_y']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        else:
            n_cells = self.params['n_inh']
            n_theta = self.params['n_theta_inh']
            n_v = self.params['n_v_inh']
            n_rf_x = self.params['n_rf_x_inh']
            n_rf_y = self.params['n_rf_y_inh']
            if n_v == 1:
                v_min = self.params['v_min_tp'] + .5 * (self.params['v_max_tp'] - self.params['v_min_tp'])
                v_max = v_min
            else:
                v_max = self.params['v_max_tp']
                v_min = self.params['v_min_tp']

        tuning_prop = np.zeros((n_cells, 5))
        if self.params['log_scale']==1:
            v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
        else:
            v_rho = np.logspace(np.log(v_min)/np.log(self.params['log_scale']),
                            np.log(v_max)/np.log(self.params['log_scale']), num=n_v,
                            endpoint=True, base=self.params['log_scale'])
        v_theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        n_orientation = self.params['n_orientation']
        orientations = np.linspace(0, np.pi, n_orientation, endpoint=False)
    #    orientations = np.linspace(-.5 * np.pi, .5 * np.pi, n_orientation)

        parity = np.arange(self.params['n_v']) % 2


        xlim = (0, self.params['torus_width'])
        ylim = (0, np.sqrt(3) * self.params['torus_height'])

        RF = np.zeros((2, n_rf_x * n_rf_y))
        X, Y = np.mgrid[xlim[0]:xlim[1]:1j*(n_rf_x+1), ylim[0]:ylim[1]:1j*(n_rf_y+1)]

        # It's a torus, so we remove the first row and column to avoid redundancy (would in principle not harm)
        X, Y = X[1:, 1:], Y[1:, 1:]
        # Add to every even Y a half RF width to generate hex grid
        Y[::2, :] += (Y[0, 0] - Y[0, 1])/2 # 1./n_RF
        RF[0, :] = X.ravel()
        RF[1, :] = Y.ravel() 
        RF[1, :] /= np.sqrt(3) # scale to get a regular hexagonal grid

        # wrapping up:
        index = 0
        random_rotation = 2*np.pi*rnd.rand(n_rf_x * n_rf_y * n_v * n_theta*n_orientation) * self.params['sigma_rf_direction']
        random_rotation_for_orientation = np.pi*rnd.rand(n_rf_x * n_rf_y * n_v * n_theta * n_orientation) * self.params['sigma_rf_orientation']

            # todo do the same for v_rho?
        for i_RF in xrange(n_rf_x * n_rf_y):
            for i_v_rho, rho in enumerate(v_rho):
                for i_theta, theta in enumerate(v_theta):
                    for orientation in orientations:
                    # for plotting this looks nicer, and due to the torus property it doesn't make a difference
                        tuning_prop[index, 0] = (RF[0, i_RF] + self.params['sigma_rf_pos'] * rnd.randn()) % self.params['torus_width']
                        tuning_prop[index, 1] = (RF[1, i_RF] + self.params['sigma_rf_pos'] * rnd.randn()) % self.params['torus_height']
                        tuning_prop[index, 2] = np.cos(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                                * rho * (1. + self.params['sigma_rf_speed'] * rnd.randn())
                        tuning_prop[index, 3] = np.sin(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                                * rho * (1. + self.params['sigma_rf_speed'] * rnd.randn())
                        tuning_prop[index, 4] = (orientation + random_rotation_for_orientation[index]) % np.pi

                        index += 1

        return tuning_prop
