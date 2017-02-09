# file I/O
import gzip
import numpy as np
import time
import pandas as pd

start_time = time.time()
'''
trainData = pd.read_csv('../256_features.csv.gz',compression="gzip")
train_y = trainData.loc[:,"gap"]
train_labels = trainData.loc[:,"smiles"]
train_x = trainData.loc[:,:]
train_x = train_x.drop("smiles",axis=1)
train_x = train_x.drop("gap",axis=1)

train_y = np.array(train_y)
train_x = np.array(train_x)
train_x = train_x[ :, train_x.any(axis=0) ]
# print("--- %s seconds ---" % (time.time() - start_time))


# save out numpy arrays to avoid recoompute
import pickle
with open('trainx.pkl', 'wb') as x:
	pickle.dump(train_x, x)
with open('trainy.pkl', 'wb') as y:
	pickle.dump(train_y, y)
'''

smiles = []
train_y = []
with gzip.open('../train.csv.gz', 'r') as fu:
	for line in fu:
		x = line.split(',')
		if x[0] == 'smiles':
			continue
		smiles.append(x[0])
		train_y.append(float(x[-1]))
train_y = np.array(train_y)
train_y.shape = (1000000,)

train_x = np.zeros((1000000, 256), dtype=bool)
with gzip.open('../256_features.csv.gz', 'r') as f:
	line_id = -1
	for line in f:
		if line_id == -1:
			line_id = line_id + 1
			continue
		x = line.split(',')
		train_x[:,line_id] = [bool(float(i)) for i in x[1:]]
		line_id = line_id + 1
		
		

# model building
import sklearn.ensemble as ens

# random k-fold validation
n = 1000 ## fold size
n_folds = 2 ## number of repetitions

# grid search parameters
n_trees = [10, 25, 50]
max_depth = [2, 6, 12]
max_features = [None]
# learning_rate = [0.05, 0.1, 0.3]

# grid search for n_estimators and max_depth (default is until accuracy is 100%)
# importances = []
for (nt, md, mfw) in [(i,j,k) for i in n_trees for j in max_depth for k in max_features]: 
	print "testing..."
	mse = []
	for trial in range(n_folds):
		# fit the model to a random subset
		sample = np.random.choice(train_x.shape[0],2*n) ## half is train, half is validation
		print "building..."
		# Create a random forest object with the n_trees and depth parameters. Will have 72 total models (3x3x8)
		clf = ens.GradientBoostingRegressor(n_estimators=nt, max_depth = md, max_features=mfw)
		print "predicting..."
		# Fit this model to the data. train_x[(the first 1000 of the 2*1000 block of validation + training data), all features], train_y label vector
		clf.fit(train_x[sample[0:n],:], train_y[sample[0:n]])

		# test and store mse predict(train_x[sample[n:-1] means that you take the second half of the 2*1000 sample (validation), all features])
		xhat = clf.predict(train_x[sample[n:-1],:])

		# get feature importances
#		importances.append(clf.feature_importances_)
#		std = np.std([tree.feature_importances_ for tree in clf.estimators_],
 #   		         axis=0)

		# take the Root Mean Squared error (diff between the actual value of train_y and the xhat predicted by the model)
		mse.append((sum([(train_y[sample[n:-1]][i] - xhat[i])**2 for i in range(n-1)])/float(n-1))**0.5)

	print '{0} Trees, {1} Depth, {3} Features: {2}'.format(nt, md, sum(mse)/float(len(mse)), mfw)
'''
importances = [np.mean([l[i] for l in importances]) for i in range(len(importances[1]))]
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(train_x.shape[1]):
	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print indices[0:50]
'''
'''
# predicting
test_values = []
with gzip.open('test.csv.gz') as g:
	for line in g:
		if line[0] == 's': ## pass over first line
			continue
		test_values.append([bool(x) for x in line.split(',')[1:-1]])

test_values = np.array(test_values, dtype=bool)'''
