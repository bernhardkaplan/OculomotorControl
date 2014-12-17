import sys
import numpy as np
import simulation_parameters
import utils
import matplotlib.pyplot as plt

if __name__ == '__main__':

    if len(sys.argv) < 2:
        GP = simulation_parameters.global_parameters()
        params = GP.params
    else:
        params = utils.load_params(sys.argv[1])


#    xlim = (0., 1.)
    xlim = (0.3, 0.7)
    vlim = (-2., 2.)
    dx = 0.05
    dv = 0.05

    xgrid = np.arange(xlim[0], xlim[1] + dx, dx)
    vgrid = np.arange(vlim[0], vlim[1] + dv, dv)
    n_x = xgrid.size
    n_v = vgrid.size
    d = np.zeros((n_x, n_v))
#    print 'n_x, n_v', n_x, n_v
#    print 'xgrid', xgrid
#    print 'vgrid', vgrid
#    print 'opt act map:', d
    for i_x in xrange(n_x):
        for i_v in xrange(n_v):
            stim_params = (xgrid[i_x], .5, vgrid[i_v], .0)
            (v_opt, vx, action_idx) = utils.get_optimal_action(params, stim_params)
            d[i_x, i_v] = action_idx

    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = d.transpose()
    CS = ax.contourf(xgrid, vgrid, d, params['n_actions'], cmap=plt.cm.jet, \
                        vmin=0, vmax=params['n_actions']-1)
    xmid = .5 * (xlim[1] - xlim[0]) + xlim[0]
    ax.plot((xmid, xmid), (vlim[0], vlim[1]), ls='--', lw=3, c='k')
    cb = plt.colorbar(CS)
    fontsize=18
    ax.set_ylabel('$v_{stim}$', fontsize=fontsize)
    ax.set_xlabel('$x_{stim}$', fontsize=fontsize)
    ax.set_title('Optimal action map', fontsize=fontsize)
    cb.set_label('Action indices', fontsize=fontsize)
    plt.show()
