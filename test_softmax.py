
import scipy.stats as stats
def softmax(a, T=1):
    a_new = np.zeros(a.size)
    exp_sum = np.sum(np.exp(T * a))
    for i_ in xrange(a.size):
        a_new[i_] = np.exp(T * a[i_]) / exp_sum
    return a_new


def draw_from_discrete_distribution(prob_dist, size=1):
    """
    prob_dist -- array containing probabilities

    E.g.
    xk = np.arange(7)
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1)
    custm = stats.rv_discrete(name='custm', values=(xk, pk))
    R = custm.rvs(size=100)
    """
    xk = np.arange(prob_dist.size)
    custm = stats.rv_discrete(name='custm', values=(xk, prob_dist))
    R = custm.rvs(size=size)
    return R


