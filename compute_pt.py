import numpy as np
import sys
from scipy.signal import find_peaks

fname = 'video'
fps = 60

# read particle motion curve and note length of video
end_video = np.loadtxt(f'stop_time_{fname}.txt')
end_video = np.convolve(end_video, np.ones(10) / 10, mode='valid')
sz = np.shape(end_video)
t = np.linspace(0, (len(end_video) * (fps / 10)) / fps, sz[0])

# calculate start time
# read start_time file
begin_video = np.loadtxt(f'start_time_{fname}.txt')

# find knee point of pipette motion curve
#def knee_pt(arr):
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

    print(x.shape)
    print(y.shape)

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

kp = knee_pt(np.convolve(begin_video, np.ones(10) / 10, mode='valid'))

# crop from start to knee point
begin_video = begin_video[:kp]
xstart = np.linspace(0, (len(begin_video) * (fps / 10)) / fps, len(begin_video))

# find most prominent peak in cropped range
# then identify start time
peaks, _ = find_peaks(begin_video)
prominent_peak = np.argmax(peaks)
startloc = prominent_peak
begin_time = t[startloc]

# calculate end time
# offset the start point by 10 seconds from when measurement starts
offset = startloc + (10 * 10)
end_video = end_video[offset:]

# normalize motion curve between [0, 1]
# trim off end of motion curve
end_video = (end_video - np.min(end_video)) / (np.max(end_video) - np.min(end_video))
f = np.where(end_video < 0.01)[0]
end_video = end_video[:]
kp = knee_pt(end_video) + offset
end_time = t[kp]

# calculate PT and INR
pt = end_time - begin_time
pt_normal = 12
isi = 1.31
alpha = -0.31
inr = (pt / pt_normal) ** (isi - alpha)
print(f'PT: {pt:.1f}\nINR: {inr:.1f}')
