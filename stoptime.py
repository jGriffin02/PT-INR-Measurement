import numpy as np
import cv2
import os
import array

def circlecropbw(img1, cy, cx, cr, invert):
    img2 = (cv2.imread(f"{fname}/frame{int(start)}.jpg", cv2.COLOR_GRAY2BGR))
    img2.astype(np.uint8)
    out, mask = circlecrop(img2, cx, cy, cr, invert)
    out = out[:, :, 0]
    return out, mask

def circlecrop(img2, cx, cy, cr, invert):
    rows, columns, numberOfColorChannels = img2.shape

    #Unsure if below should have "...columns, 3)" or "..., 2)" ==> its for color channels, which should be 3, starts at 0 or 1?
    rgbImage2 = np.zeros((rows, columns, 3), dtype=np.uint8)

    #Does this cv2.resize break the code? Need # of color channels
    rgbImage2 = cv2.resize(rgbImage2, (columns, rows))

    ci = [cx, cy, cr]

    [xx, yy] = np.meshgrid(np.arange(columns) - ci[1], np.arange(rows) - ci[2])
    mask = (np.power(xx,2) + np.power(yy,2) < (ci[2]**2))

    if invert == 1:
        mask = ~mask

    redChannel1 = img2[:, :, 2]
    greenChannel1 = img2[:, :, 1]
    blueChannel1 = img2[:, :, 0]

    #These below are errors, what should happen is taking RGB channels of rgbImage2
    redChannel2 = rgbImage2[:, :, 2]
    greenChannel2 = np.copy(rgbImage2[:, :, 1])
    blueChannel2 = np.copy(rgbImage2[:, :, 0])

    redChannel2[mask] = redChannel1[mask]
    greenChannel2[mask] = greenChannel1[mask]
    blueChannel2[mask] = blueChannel1[mask]

    out = cv2.merge([blueChannel2, greenChannel2, redChannel2])

    return out, mask


#height, width, _ = img.shape
    #mask = np.zeros((height, width), dtype=np.uint8)

#    cv2.circle(mask, (int(cx), int(cy)), int(cr), 255, -1)

  #  if invert:
 #       mask = cv2.bitwise_not(mask)

   # out = cv2.bitwise_and(img, img, mask=mask)

    #return out, mask


y = 920
x = 500
r = 200
fps = 60

fname = 'video'
n = len([name for name in os.listdir(fname) if os.path.isfile(os.path.join(fname, name))]) - 3
n = int(n / fps) * fps - (fps / 10)

met = []
for start in np.arange(0, n, fps / 10):
    img1 = cv2.imread(f"{fname}/frame{int(start)}.jpg", cv2.IMREAD_ANYCOLOR)

    #Shows the image being used
    #while True:
    #    cv2.imshow(f"{fname}/frame{int(start)}.jpg", img1)
    #    cv2.waitKey(0)
    #    sys.exit()
    #cv2.destroyAllWindows()

    sz = np.asarray(np.shape(img1))
    if sz[0] == 1080:
        #                                                      Added np.asarray( ) to change later img1 processing L104
        img1 = np.asarray(np.rot90(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 3))
    img1 = np.asarray(circlecropbw(img1, y, x, r, 0))

    #                                                          Added below to set img1 as 8 bit image
    img1.astype(np.uint8)

    #Original > Changed
    #img1 = img1[y - r:y + r, x - r:x + r, :]

    #Array of num from y-r ==> y+r && x-r ==> x+r
    #array(slice_obj[::1])

    arr1 = []
    add = 0
    for i in range(y-r, y+r, 1):
        arr1.append((y - r) + add)
        add = add + 1
    arr1.append(y + r)
    arr1 = np.array(arr1)

    arr2 = []
    add = 0
    for i in range(x-r, x+r, 1):
        arr2.append((x-r)+add)
        add = add + 1
    arr2.append(x + r)
    arr2 = np.array(arr2)


    print(arr1.dtype)

    if img1.size == 0:
        print("img1 empty")

    img1 = img1[arr1, :][:, arr2, 0]

    #img1 = img1.item([arr1, arr2])
    quit()

    img2 = cv2.imread(f"{fname}/frame{int(start + (fps / 10))}.jpg")
    sz = np.shape(img2)
    if sz[0] == 1080:
        img2 = np.rot90(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 3)
    img2 = circlecropbw(img2, y, x, r, 0)

    #Original
    #img2 = img2[y - r:y + r, x - r:x + r, :]



    cc = np.abs(img1 - img2)
    met.append(np.sum(cc))
    print(str((start / n) * 100) + '% complete')
np.savetxt(f"stop_time_{fname}.txt", met)
