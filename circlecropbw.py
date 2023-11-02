import cv2
import numpy as np

def circlecrop(img, cx, cy, cr, invert=False):
    #gets rows, columns, and # of color channels from input image
    rows, cols, numberOfColorChannels = img.shape

    #creates variable rgbImg2 and sets it equal to shape of (rows, cols) w/ 3 colorchannels,
    #   sets type to unsigned int(max=255)
    rgbImg2 = np.zeros((rows, cols, 3), dtype=np.uint8)

    #resizes rgbImg2 to size of (cols, rows)
    cv2.resize(rgbImg2, (cols, rows))

    #sets new variable ImageSize to same shape as input image
    ImageSize = img.shape

    #Creates vector/array ci and sets = [cx, cy, cr]
    ci = [cx, cy, cr]

    #Creates xx, yy, and sets them equal to
    xx, yy = np.meshgrid(np.arrange(ImageSize[1]) - ci[0], np.arrange(ImageSize[0]) - ci[1])

    #Creates mask, sets equal to " (xx squared + yy squared) {is less than} vector.ci[2] squared "
    mask = (xx**2 + yy**2) < ci[2]**2

    #creates circle in input, at (cx, cy), with radius 'cr', with color ( ), and line thickness
    cv2.circle(mask, (cx, cy), cr, (255,255,255),-1)

    if invert:
        mask = 255-mask

    #
    out = np.zeros_like(img)

    #
    out = cv2.bitwise_and(img, mask)

    return out, mask

def circlecropbw(img, cx, cy, cr, invert=False):
    out, mask = circlecrop(img, cx, cy, cr, invert)
    out = out[:, :, 0]
    return out, mask