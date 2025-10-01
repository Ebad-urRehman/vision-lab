import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('image.png')

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# scaling
scaled = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)

# Translation
rows, cols = img.shape[:2]

tr = np.float32([[1,0,50], [0,1,30]])
translated = cv.warpAffine(img, tr, (cols, rows))

# roation
center = (cols//2, rows//2)

M_rotate = cv.getRotationMatrix2D(center, 45, 1.0) # matrix roation by 45 degree
rotated_image = cv.warpAffine(img, M_rotate, (cols, rows))

# flipiing 
flip_x = cv.flip(img, 0)
flip_y = cv.flip(img, 1)
flip_xy = cv.flip(img, -1)


plt.figure(figsize=(10,15))

plt.subplot(2,2,1)
plt.axis('off')
plt.title('Rotated Image')

plt.imshow(rotated_image)


plt.subplot(2,2,2)
plt.axis('off')
plt.title('Scaled Image')
plt.imshow(scaled)


plt.subplot(2,2,3)
plt.axis('off')
plt.title('Translated Image')
plt.imshow(translated)


plt.subplot(2,2,4)
plt.axis('off')
plt.title('Original Image')
plt.imshow(img_rgb)


plt.show()


plt.figure(figsize=(10,15))

plt.subplot(2,2,1)
plt.axis('off')
plt.title('Original Image')
plt.imshow(img_rgb)

plt.subplot(2,2,2)
plt.axis('off')
plt.title('Flip X Image')
plt.imshow(flip_x)

plt.subplot(2,2,3)
plt.axis('off')
plt.title('Flip Y Image')
plt.imshow(flip_y)

plt.subplot(2,2,4)
plt.axis('off')
plt.title('Flip XY Image')
plt.imshow(flip_xy)

plt.show()