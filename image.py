from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from ikrlib import png2fea
import scipy.linalg
import numpy as np


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


train = {}
test = {}

average = np.zeros((80, 80), dtype=np.float64).reshape(-1)

train_set_names = []
train_set_list = []

test_set_names = []
test_set_list = []

num = 20 

img_counter = 0

for i in range(1, 32):
    train[i] = png2fea("./train/" + str(i))
    for key in train[i]:
        tmp = train[i][key].reshape(-1)
        train_set_list.append(tmp)
        train_set_names.append(i)
        average += tmp
        img_counter += 1
        

average = average / float(img_counter)

#plt.imshow(average.reshape(80, 80), cmap="gray")
#plt.show()

test_counter = 0

for i in range(1, 32):
    test[i] = png2fea("./dev/" + str(i))
    for key in test[i]:
        tmp = test[i][key].reshape(-1)
        test_set_list.append(tmp)
        test_set_names.append(i)
        test_counter += 1


#normalize
for i in range(len(train_set_list)):
    train_set_list[i] = train_set_list[i] - average

train_matrix = np.array(train_set_list)

for i in range(len(test_set_list)):
    test_set_list[i] = test_set_list[i] - average

test_matrix = np.array(test_set_list)

###### PCA of train

#big_matrix = np.concatenate((train_matrix, test_matrix), axis=0)
 
# C = AA' --> python C = A'A
#cov_matrix = np.cov(big_matrix)
#[evals, evect] = np.linalg.eig(cov_matrix)

#sort eigen vectors
#sorted_args = np.argsort(evals)
#sorted_args = sorted_args[::-1]
#sorted_vals = np.sqrt(np.sort(evals))
#sorted_vect = evect[:,sorted_args]

pca = PCA(n_components=num, svd_solver='randomized').fit(train_matrix)

train_vectors = pca.transform(train_matrix)
#sorted_vect[:,range(num)]

#mapping = None
#cutting last low signifant eigen_faces
#if num < size and num >= 0:
#    sorted_vals = sorted_vals[range(num)]
#    sorted_vect = sorted_vect[:,range(num)]
#    mapping = np.dot(sorted_vect.T, train_matrix)
#else:
#    print "Num of eigen faces is grater than size\n"
#    exit()


###### PCA of test

#cov_matrix = np.cov(test_matrix)
#[evals, evect] = np.linalg.eig(cov_matrix)

#sorted_args = np.argsort(evals)
#sorted_args = sorted_args[::-1]
#sorted_vals = np.sqrt(np.sort(evals))
#sorted_vect = evect[:,sorted_args]

test_vectors = pca.transform(test_matrix)

clf = SVC(kernel="poly")
clf.fit(train_vectors, train_set_names)

print clf.predict(test_vectors)
print clf.score(test_vectors, test_set_names)

#clf.predict_proba(...)
#clf.predictlog_proba(....)



#sorted_vect[:,range(num)]

###### computing distancies

#results = np.zeros((test_counter, img_counter), dtype=np.float64)

#for index, train_vector in enumerate(train_vectors):
#    if index >= len(train_set_list):
#        break
#    for i, test_vector in enumerate(test_vectors):
#        dist = np.linalg.norm(train_vector - test_vector)
#        results[i][index] = dist
#        print str((index / 6) + 1) + "/" + str((index % 6) + 1) + " vs " + str((i / 2) + 1) + "/" + str((i % 2) + 1) + " --> " + str(dist) 

#print results

#evaluate = {}

#for i in range(len(results)):
#    index = train_set_names[np.argmin(results[i])]
#    value = np.min(results[i])
#    evaluate[i] = (index, value)

#counter = 0
#for i in range(len(evaluate)):
#    s = ""
#    if (i / 2) != evaluate[i][0]:
#        s = "-->miss<-- "
 #   else:
 ##       s = "-->ok<-- "
 #       counter += 1
 #   print str(i+1) + " --> guess: " + str(evaluate[i][0]) + " reality: " + str((i/2) + 1) + " " + s + str(evaluate[i][1])

#print "counter: " + str(counter)  

#eigen_faces = np.dot(sorted_vect, mapping) - train_matrix

#for i in range(0, num):
#    eigen_faces[:,i] /= sorted_vals[i] 
#
#for i in range(10):
#    plt.figure()
#    plt.imshow(eigen_faces[i].reshape(80,80),  cmap="gray")
#
#plt.show()



 

# imgs = np.concatenate([(img + average).reshape(80,80) for img in train_set_list], axis=1)
#plt.imshow(imgs,  cmap="gray")
#plt.show()

#plt.imshow((train_set[10] + average).reshape(80, 80), cmap="gray")
#plt.show()








        

#plt.imshow(train[1]["./train/1/f401_02_f18_i0_0.png"].astype(int), cmap="gray")
#plt.show()

