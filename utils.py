"""
    This file contains a bunch of helper functions (in alphabetic order).
"""

import numpy as np
import numpy.random as rnd
import os


def set_tuning_prop(params, mode, cell_type):
    if params['n_grid_dimensions'] == 2:
        return set_tuning_prop_2D(params, mode, cell_type)
    else:
        return set_tuning_prop_1D(params, cell_type)


def set_tuning_prop_1D(params, cell_type='exc'):

    rnd.seed(params['tuning_prop_seed'])
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_v = params['n_v']
        n_rf_x = params['n_rf_x']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_v = params['n_v_inh']
        n_rf_x = params['n_rf_x_inh']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    tuning_prop = np.zeros((n_cells, 5))
    if params['log_scale']==1:
        v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
    else:
        v_rho = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v,
                        endpoint=True, base=params['log_scale'])
    n_orientation = params['n_orientation']
    orientations = np.linspace(0, np.pi, n_orientation, endpoint=False)
    xlim = (0, params['torus_width'])

    RF = np.linspace(0, params['torus_width'], n_rf_x, endpoint=False)
    index = 0
    random_rotation_for_orientation = np.pi*rnd.rand(n_rf_x * n_v * n_orientation) * params['sigma_rf_orientation']

        # todo do the same for v_rho?
    for i_RF in xrange(n_rf_x):
        for i_v_rho, rho in enumerate(v_rho):
            for orientation in orientations:
            # for plotting this looks nicer, and due to the torus property it doesn't make a difference
                tuning_prop[index, 0] = (RF[i_RF] + params['sigma_rf_pos'] * rnd.randn()) % params['torus_width']
                tuning_prop[index, 1] = 0.5 # i_RF / float(n_rf_x) # y-pos 
                tuning_prop[index, 2] = rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                tuning_prop[index, 3] = 0. # np.sin(theta + random_rotation[index]) * rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                tuning_prop[index, 4] = (orientation + random_rotation_for_orientation[index]) % np.pi

                index += 1

    return tuning_prop




def set_tuning_prop_2D(params, mode='hexgrid', cell_type='exc'):
    """
    Place n_exc excitatory cells in a 4-dimensional space by some mode (random, hexgrid, ...).
    The position of each cell represents its excitability to a given a 4-dim stimulus.
    The radius of their receptive field is assumed to be constant (TODO: one coud think that it would depend on the density of neurons?)

    return value:
        tp = set_tuning_prop(params)
        tp[:, 0] : x-position
        tp[:, 1] : y-position
        tp[:, 2] : u-position (speed in x-direction)
        tp[:, 3] : v-position (speed in y-direction)

    All x-y values are in range [0..1]. Positios are defined on a torus and a dot moving to a border reappears on the other side (as in Pac-Man)
    By convention, velocity is such that V=(1,0) corresponds to one horizontal spatial period in one temporal period.
    This implies that in one frame, a translation is of  ``1. / N_frame`` in cortical space.
    """

    rnd.seed(params['tuning_prop_seed'])
    if cell_type == 'exc':
        n_cells = params['n_exc']
        n_theta = params['n_theta']
        n_v = params['n_v']
        n_rf_x = params['n_rf_x']
        n_rf_y = params['n_rf_y']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
    else:
        n_cells = params['n_inh']
        n_theta = params['n_theta_inh']
        n_v = params['n_v_inh']
        n_rf_x = params['n_rf_x_inh']
        n_rf_y = params['n_rf_y_inh']
        if n_v == 1:
            v_min = params['v_min_tp'] + .5 * (params['v_max_tp'] - params['v_min_tp'])
            v_max = v_min
        else:
            v_max = params['v_max_tp']
            v_min = params['v_min_tp']

    tuning_prop = np.zeros((n_cells, 5))
    if params['log_scale']==1:
        v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
    else:
        v_rho = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                        np.log(v_max)/np.log(params['log_scale']), num=n_v,
                        endpoint=True, base=params['log_scale'])
    v_theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    n_orientation = params['n_orientation']
    orientations = np.linspace(0, np.pi, n_orientation, endpoint=False)
#    orientations = np.linspace(-.5 * np.pi, .5 * np.pi, n_orientation)

    parity = np.arange(params['n_v']) % 2


    xlim = (0, params['torus_width'])
    ylim = (0, np.sqrt(3) * params['torus_height'])

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
    random_rotation = 2*np.pi*rnd.rand(n_rf_x * n_rf_y * n_v * n_theta*n_orientation) * params['sigma_rf_direction']
    random_rotation_for_orientation = np.pi*rnd.rand(n_rf_x * n_rf_y * n_v * n_theta * n_orientation) * params['sigma_rf_orientation']

        # todo do the same for v_rho?
    for i_RF in xrange(n_rf_x * n_rf_y):
        for i_v_rho, rho in enumerate(v_rho):
            for i_theta, theta in enumerate(v_theta):
                for orientation in orientations:
                # for plotting this looks nicer, and due to the torus property it doesn't make a difference
                    tuning_prop[index, 0] = (RF[0, i_RF] + params['sigma_rf_pos'] * rnd.randn()) % params['torus_width']
                    tuning_prop[index, 1] = (RF[1, i_RF] + params['sigma_rf_pos'] * rnd.randn()) % params['torus_height']
                    tuning_prop[index, 2] = np.cos(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                            * rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                    tuning_prop[index, 3] = np.sin(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                            * rho * (1. + params['sigma_rf_speed'] * rnd.randn())
                    tuning_prop[index, 4] = (orientation + random_rotation_for_orientation[index]) % np.pi

                    index += 1

    return tuning_prop
