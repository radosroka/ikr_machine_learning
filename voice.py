import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint

train = {}
MUs = {}
COVs = {}
gauss_cnt = 10
Ws = {}
for i in range(1, 32):
	
	train[i] = wav16khz2mfcc('train/' + str(i)).values()

	#plt.plot(train[i][0][:,0])
	#plt.show()

	# DELETE FIRST AND LAST 200 FRAMES
	for j in range(0, 6):
		for k in range(0, 200):
			train[i][j] = np.delete(train[i][j], (0), axis=0)
		for k in range(0, 200):
			train[i][j] = np.delete(train[i][j], (len(train[i][j])-1), axis=0)

	# DELETE SILENCE frames
	for j in range(0, 6):
		summ = 0
		min_eng = train[i][j][0][0]
		for k in range(0, len(train[i][j])):
			summ = summ + train[i][j][k][0]
			if (train[i][j][k][0] < min_eng):
				min_eng = train[i][j][k][0]

		avg_eng = summ / len(train[i][j])

		treshold = (min_eng + avg_eng) / 2		

		indices = []
		low_eng_cnt = 0
		for k in range(0, len(train[i][j])):
			if (train[i][j][k][0] < treshold):
				low_eng_cnt += 1
			else:
				low_eng_cnt = 0

			if (low_eng_cnt >= 60):
				low_eng_cnt = 0
				for m in range(k-59, k+1):
					indices.append(m)
	
		train[i][j] = np.delete(train[i][j], indices, axis=0)
		
		summ = 0
		for k in range(0, len(train[i][j])):
			summ = summ + train[i][j][k][0]

		avg_eng = summ / len(train[i][j])

		#plt.plot(train[i][0][:,0])
		#plt.show()

                # normalize the signal
		for k in range(0, len(train[i][j])):
			train[i][j][k][0] = train[i][j][k][0] - avg_eng

		#plt.plot(train[i][0][:,0])
		#plt.show()


	train[i] = np.vstack(train[i])

	MUs[i] = (train[i][randint(1, len(train[i]), gauss_cnt)])
	COVs[i] = ([np.var(train[i], axis=0)] * gauss_cnt)
	Ws[i] = np.ones(gauss_cnt) / gauss_cnt;

	for j in range(25):
		[Ws[i], MUs[i], COVs[i], TTL] = train_gmm(train[i], Ws[i], MUs[i], COVs[i]);
		print('Iteration:', j, ' Total log-likelihood:', TTL, 'for class ' + str(i))


score = []
for i in range(1, 32):

	test = wav16khz2mfcc('dev/' + str(i)).values()

	for j in range(0, 2):
		for k in range(0, 200):
			test[j] = np.delete(test[j], (0), axis=0)
		for k in range(0, 200):
			test[j] = np.delete(test[j], (len(test[j])-1), axis=0)

	for j in range(0, 2):
		summ = 0
		min_eng = test[j][0][0]
		for k in range(0, len(test[j])):
			summ = summ + test[j][k][0]
			if (test[j][k][0] < min_eng):
				min_eng = test[j][k][0]

		avg_eng = summ / len(test[j])

		treshold = (min_eng + avg_eng) / 2

		indices = []
		low_eng_cnt = 0
		for k in range(0, len(test[j])):
			if (test[j][k][0] < treshold):
				low_eng_cnt += 1
			else:
				low_eng_cnt = 0

			if (low_eng_cnt >= 60):
				low_eng_cnt = 0
				for m in range(k-59, k+1):
					indices.append(m)

		train[j] = np.delete(test[j], indices, axis=0)

		summ = 0
		for k in range(0, len(test[j])):
			summ = summ + test[j][k][0]

		avg_eng = summ / len(test[j])

		#plt.plot(train[i][0][:,0])
		#plt.show()

		for k in range(0, len(test[j])):
			test[j][k][0] = test[j][k][0] - avg_eng

	for tst in test:

		ll = []
		for j in range(1, 32):
			ll.append(sum(logpdf_gmm(tst, Ws[j], MUs[j], COVs[j])))


		score.append(i == (np.argmax(ll) + 1))

print(np.average(score))
