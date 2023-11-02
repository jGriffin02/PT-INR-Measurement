import random
import sys

import numpy as np
import array as arr
from scipy.stats import linregress

def knee_pt(y, x=None, just_return=False):
    use_absolute_dev_p = False
    issue_errors_p = True

    if just_return:
        issue_errors_p = False

    res_x = np.nan
    idx_of_result = np.nan

    if len(y) == 0:
        if issue_errors_p:
            raise ValueError('knee_pt: y can not be an empty vector')
        return res_x, idx_of_result

    y = np.vstack(np.array(y).flatten())

    if x is None or len(x) == 0:
        x = np.matrix(np.arange(1, len(y) + 1)).getH()
    else:
        x = np.matrix(x).getH()

    if x.shape != y.shape or np.matrix(x).shape != y.shape:
        if issue_errors_p:
            raise ValueError('knee_pt: y and x must have the same dimensions')
        return res_x, idx_of_result

    if len(y) < 3:
        if issue_errors_p:
            raise ValueError('knee_pt: y must be at least 3 elements long')
        return res_x, idx_of_result

    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        y = y[idx]
        x = x[idx]
    else:
        idx = np.arange(1, len(x) + 1)


    sigma_xy = np.cumsum(np.multiply(x, y))
    sigma_x = np.cumsum(x)
    sigma_y = np.cumsum(y)
    sigma_xx = np.cumsum(np.multiply(x, x))
    n = np.matrix(np.arange(1, len(y) + 1)).getH()
    det = (np.squeeze(np.asarray(n)) * np.squeeze(np.asarray(sigma_xx))) \
          - (np.squeeze(np.asarray(sigma_x)) * np.squeeze(np.asarray(sigma_x)))

    mfwd = ((np.squeeze(np.asarray(n)) * np.squeeze(np.asarray(sigma_xy))) \
           - (np.squeeze(np.asarray(sigma_x)) * np.squeeze(np.asarray(sigma_y)))) / det

    bfwd = -((np.squeeze(np.asarray(sigma_x)) * np.squeeze(np.asarray(sigma_xy))) \
           - (np.squeeze(np.asarray(sigma_xx)) * np.squeeze(np.asarray(sigma_y)))) / det

    sigma_xy = np.cumsum(np.multiply(x[::-1], y[::-1]))
    sigma_x = np.cumsum(x[::-1])
    sigma_y = np.cumsum(y[::-1])
    sigma_xx = np.cumsum(np.multiply(x[::-1], x[::-1]))
    det = (np.squeeze(np.asarray(n)) * np.squeeze(np.asarray(sigma_xx))) \
          - (np.squeeze(np.asarray(sigma_x)) * np.squeeze(np.asarray(sigma_x)))
    n = np.matrix(np.arange(1, len(y) + 1)).getH()

    y = y[::-1]
    x = x[::-1]

    mbck = (np.flipud(np.divide((np.squeeze(np.asarray(n)) * np.squeeze(np.asarray(sigma_xy)))
                     - (np.squeeze(np.asarray(sigma_x)) * np.squeeze(np.asarray(sigma_y))) , det)))

    bbck = np.flipud(-((np.squeeze(np.asarray(sigma_x)) * np.squeeze(np.asarray(sigma_xy)))
                       - (np.squeeze(np.asarray(sigma_xx)) * np.squeeze(np.asarray(sigma_y)))) / det)

    error_curve = np.zeros_like(y, dtype=float)
    error_curve[:] = np.nan

    for breakpt in range(1, len(y - 1)):
        delsfwd = (np.squeeze(np.asarray(mfwd[breakpt])) * np.squeeze(np.asarray(x[::-1][:breakpt + 1]))
                   + bfwd[breakpt]) - np.squeeze(np.asarray(y[::-1][:breakpt + 1]))

        delsbck = (np.squeeze(np.asarray(mbck[breakpt])) * np.squeeze(np.asarray(x[::-1][breakpt:]))
                   + bbck[breakpt]) - np.squeeze(np.asarray(y[::-1][breakpt:]))

        if use_absolute_dev_p:
            error_curve[breakpt] = np.sum(np.abs(delsfwd)) + np.sum(np.abs(delsbck))

        else:
            error_curve[breakpt] = np.squeeze(np.sqrt(np.squeeze(np.sum(np.squeeze(np.asarray(delsfwd)) * np.squeeze(np.asarray(delsfwd)))))
                                              + np.sqrt(np.squeeze(np.sum(np.squeeze(np.asarray(delsbck)) * np.squeeze(np.asarray(delsbck))))))

    loc = np.nanargmin(error_curve)
    res_x = x[loc]
    idx_of_result = idx[loc]

    return res_x, idx_of_result

print("we did it")
print(knee_pt(np.matrix([30, 27, 24, 21, 18, 15, 12, 10, 8, 6, 4, 2, 0])))