import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# brightness make color combination of every thing higher
# contrast make dark and light regions prominent

img = cv.imread('image.png', 0)

alpha = 1.4
beta = 30
bright_contrast = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

# log transformation applied majorly on medical images 
img_float = img.astype(np.float32) + 1

c = 255 / np.log(1 + img_float) # scaling factor

log_img = c * np.log(img_float)

log_img = np.clip(log_img, 0, 255).astype(np.uint8)

# power law or gamma transformation
gemma = 0.6


# image normalization
img_norm = img / 255.0
gemma_img = np.power(img_norm, gemma)

gemma_img = np.clip(gemma_img, 0, 255).astype(np.uint8)


plt.figure(figsize=(10,15))

plt.subplot(2,2,1)
plt.axis('off')
plt.title('Original Image')
plt.imshow(img)

plt.subplot(2,2,2)
plt.axis('off')
plt.title('Log Scaled Image')
plt.imshow(log_img)

plt.show()

