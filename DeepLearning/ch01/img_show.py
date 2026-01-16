import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./DeepLearning/ch01/Glass.jpg')

plt.imshow(img)
plt.show()
