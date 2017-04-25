import matplotlib.pyplot as plt
from ikrlib import png2fea
import scipy.linalg
import numpy as np
from numpy.random import randint

train = {}

average = np.zeros((80, 80), dtype=np.float64)
img_counter = int(0)

for i in range(1, 31):
    train[i] = png2fea("./train/" + str(i))
    for key in train[i]:
        average += train[i][key]
        img_counter += 1

average = average / float(img_counter)

#plt.imshow(average.astype(int), cmap="gray")
#plt.show()






        

#plt.imshow(train[1]["./train/1/f401_02_f18_i0_0.png"].astype(int), cmap="gray")
#plt.show()

