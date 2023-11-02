import cv2
import numpy as np
import os
import scipy



def circlecropbw(img, y, x, r, value):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), r, (255), -1)
    cropped_img = cv2.bitwise_and(img, img, mask=mask)

    if value == 0:
        cropped_img[np.where(mask == 255)] = [0, 0, 0]

    return cropped_img

def starttime():
    y = 920
    x = 500
    r1 = 200
    r2 = 350
    fps = 60
    fname = 'video'
    n = len([name for name in os.listdir(fname) if os.path.isfile(os.path.join(fname, name))]) - 3
    n = fps * 10
    met = []
    for start in np.arange(0, n, fps / 10):
        img1 = cv2.imread(fname + '/frame' + str(int(start)) + '.jpg')
        sz = img1.shape[:2]
        if sz[0] == 1080:
            img1 = np.rot90(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 3)
        img1 = circlecropbw(img1, y, x, r1, 1)
        img1 = img1[y - r2:y + r2, x - r2:x + r2, :]
        img2 = cv2.imread(fname + '/frame' + str(int(start + fps / 10)) + '.jpg')
        sz = img2.shape[:2]
        if sz[0] == 1080:
            img2 = np.rot90(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 3)
        img2 = circlecropbw(img2, y, x, r1, 1)
        img2 = img2[y - r2:y + r2, x - r2:x + r2, :]
        cc = np.abs(img1 - img2)
        met.append(np.sum(cc))
        print(str((start / n) * 100) + '% complete')
    np.savetxt('start_time_' + fname + '.txt', met)

def stoptime():
    def circlecropbw(img, y, x, r, value):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (x, y), r, (255), -1)
        cropped_img = cv2.bitwise_and(img, img, mask=mask)

        if value == 0:
            cropped_img[np.where(mask == 255)] = [0, 0, 0]

        return cropped_img

    y = 920
    x = 500
    r = 200
    fps = 60

    fname = 'video'
    n = len([name for name in os.listdir(fname) if os.path.isfile(os.path.join(fname, name))]) - 3
    n = int(n / fps) * fps - (fps / 10)

    met = []
    for start in np.arange(0, n, fps / 10):
        img1 = cv2.imread(f"{fname}/frame{int(start)}.jpg")
        sz = np.shape(img1)
        if sz[0] == 1080:
            img1 = np.rot90(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 3)
        img1 = circlecropbw(img1, y, x, r, 0)
        img1 = img1[y - r:y + r, x - r:x + r, :]
        img2 = cv2.imread(f"{fname}/frame{int(start + (fps / 10))}.jpg")
        sz = np.shape(img2)
        if sz[0] == 1080:
            img2 = np.rot90(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 3)
        img2 = circlecropbw(img2, y, x, r, 0)
        img2 = img2[y - r:y + r, x - r:x + r, :]
        cc = np.abs(img1 - img2)
        met.append(np.sum(cc))
        print(str((start / n) * 100) + '% complete')
    np.savetxt(f"stop_time_{fname}.txt", met)

def compute_pt():
    import numpy as np
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
    def knee_pt(arr):
        diff = arr[1:] - arr[:-1]
        return np.argmax(diff) + 1

    kp = knee_pt(np.convolve(begin_video, np.ones(10) / 10, mode='valid'))

    print(kp)
    quit()


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


if __name__ == '__main__':
    starttime()