import numpy as np
from scipy.stats import norm

def confidence(r, n, alpha = 0.05):
    z_criterion = norm.ppf(1-(alpha/2))
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    upper = np.tanh(z + z_criterion * se)
    lower = np.tanh(z - z_criterion * se)

    return lower, upper

def significance(r1, r2, n1, n2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    S = np.sqrt(1/(n1-3) + 1/(n2-3))
    return (1-norm.cdf(np.abs(z1-z2)/S)) * 2