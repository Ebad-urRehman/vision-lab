# Warp affine task

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

image = cv.imread("image.png")

image[100, 50] = (0,0,255) # set red color to pixel at x=100 and y=50 axis
image = cv.circle(image, (50, 100), 5,(0, 255, 255), 1)

image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)


plt.figure(figsize=(10, 5))


plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.axis('off')

# new zoomed
zoom = image_rgb[0:200, 0:100]

plt.subplot(1,2,2)
plt.imshow(zoom)
plt.axis('off')
plt.show()

# scaling is to change size wrt other size

# scale =  size * fx and fy values fx and fy are scaling factors
scaled = cv.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)


# translation
rows, cols = image_rgb.shape[:2]

tr = np.float32([[1, 0, 50], [0, 1, 30]])
# first diagnal sharing 2nd scaling 30 and 50 controls transformation
# 50 and 30 right and down shift
# warpaffine is operation on matrices
transformated_image = cv.warpAffine(image_rgb, tr, (cols, rows))

