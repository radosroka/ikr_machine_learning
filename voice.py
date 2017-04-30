import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
from glob import glob
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
	for j in range(0, len(train[i])):
		for k in range(0, 200):
			train[i][j] = np.delete(train[i][j], (0), axis=0)
		for k in range(0, 200):
			train[i][j] = np.delete(train[i][j], (len(train[i][j])-1), axis=0)

	# DELETE SILENCE frames
	for j in range(0, len(train[i])):
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
		#print('Iteration:', j, ' Total log-likelihood:', TTL, 'for class ' + str(i))

final = ''
score = []
for i in range(1, 2):

	

	dir_test = 'dev/' + str(i)
	dir_eval = 'eval/'

	f = glob(dir_eval + '/*.wav')
	for x in range (0, len(f)):
		f[x] = f[x][:-4]
		f[x] = f[x].replace('eval\\' , '')
	

		
		
	test = wav16khz2mfcc(dir_eval).values()
	#test = wav16khz2mfcc(dir_test).values()


	for j in range(0, len(test)):
		for k in range(0, 200):
			test[j] = np.delete(test[j], (0), axis=0)
		for k in range(0, 200):
			test[j] = np.delete(test[j], (len(test[j])-1), axis=0)

	for j in range(0, len(test)):
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

	cnt = 0
	
	for tst in test:
		
		ll = []
		for j in range(1, 32):
			ll.append(sum(logpdf_gmm(tst, Ws[j], MUs[j], COVs[j])))

		
		final += str(f[cnt])
		final += ' '
		final += str(np.argmax(ll) +1)
		final += ' '
		for z in range(1,32):
			final += str((sum(logpdf_gmm(tst, Ws[z], MUs[z], COVs[z]))))
			final += ' '
		
		final += '\n'

		cnt = cnt +1
		#score.append(i == (np.argmax(ll) + 1))

		
		

output = 'output_voice'
write = open(output , 'w')
write.write(final)
write.close()

#print(np.average(score))
