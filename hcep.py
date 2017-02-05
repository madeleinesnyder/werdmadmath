# file I/O
import gzip
import numpy as np

train = []
values = []
strings = []
with gzip.open('../train.csv.gz') as f:
	for line in f:
		if line[0] == 's': ## pass over first line
			continue
		zzz = line.split(',')
		train.append([bool(x) for x in zzz[1:-1]])
		values.append(float(zzz[-1]))
		strings.append(zzz[0])

train_x = np.array(train, dtype=bool)
train_y = np.array(values)

'''
# save out numpy arrays to avoid recoompute
import pickle
with open('trainx.pkl', 'wb') as x:
	pickle.dump(train_x, x)
with open('trainy.pkl', 'wb') as y:
	pickle.dump(train_y, y)
'''

# model building
import sklearn.ensemble as ens

# random k-fold validation
n = 10000 ## fold size
k = 8 ## number of repetitions

# grid search parameters
n_trees = [20, 100, 500]
max_depth = [None, 10, 50]

# grid search for n_estimators and max_depth (default is until accuracy is 100%)
for (nt, md) in [(i,j) for i in n_trees for j in max_depth]: 
	print nt, md
	mse = []
	for trial in range(k):
		# fit the model to a random subset
		sample = np.random.choice(train_x.shape[0],2*n) ## half is train, half is validation
		clf = ens.RandomForestRegressor(n_estimators=nt, n_jobs=-1, max_depth = md, warm_start=True)
		clf.fit(train_x[sample[0:n],:], train_y[sample[0:n]])
		# test and store mse
		xhat = clf.predict(train_x[sample[n:-1],:])
		mse.append(sum([(train_y[sample[n:-1]][i] - xhat[i])**2 for i in range(n-1)])/float(n-1))
	print '{0} Trees, {1} Depth: {2}'.format(nt, md, sum(mse)/float(len(mse)))


'''
# predicting
test_values = []
with gzip.open('test.csv.gz') as g:
	for line in g:
		if line[0] == 's': ## pass over first line
			continue
		test_values.append([bool(x) for x in line.split(',')[1:-1]])

test_values = np.array(test_values, dtype=bool)'''
