import numpy as np
import pandas as pd
import time
from sklearn import linear_model as lm
from matplotlib import pyplot as plt

start_time = time.time()

zero_idx = []
allZeros = []
mse = []

trainData = pd.read_csv('../train.csv.gz',compression="gzip")
train_y = trainData.loc[:,"gap"]
train_x = trainData.loc[:,:]
train_x = train_x.drop("smiles",axis=1)
train_x = train_x.drop("gap",axis=1)

# convert pandas to numpy arrays and drop null columns 
train_y = np.array(train_y) # 1,000,000 x 1
train_x = np.array(train_x) # 1,000,000 x 257
train_x = train_x[ :, train_x.any(axis=0) ] # 1,000,000 x 31


# zero_idx = train_x.apply(lambda x: np.all(x==0))
# for i, value in enumerate(zero_idx):
# 	if value == True:
# 		allZeros.append(i)

# Drop all the useless columns from the data matrix
# for m in range(1000000):
# 	if (train_x.iloc[m,1] == 1.0):
# 		print('no')

# print(train_x.loc[:,1].name)

# train_x.drop(train_x.loc[:,[i for i in allZeros]].name,axis=1)


print("--- %s seconds ---" % (time.time() - start_time))

# Make the k-fold thingy
k = 8 # number folds
n = 10000 # fold size
mse = []
training_values = []

for trial in range(k):
	sample = np.random.choice(train_x.shape[0],2*n) # choose 2000 random values
	reg = lm.LinearRegression()
	# fit the model to X (first 10000 rows and all colums of the random sample from train_x) and y (same 10000 rows from the labeled dataset)
	reg.fit(train_x[sample[0:n],:],train_y[sample[0:n]])
	# predict using the validation data in the model and get mse
	xhat = reg.predict(train_x[sample[n:-1],:]) 
	# append mse
	mse.append(sum([(train_y[sample[n:-1]][i] - xhat[i])**2 for i in range(n-1)])/float(n-1))
	training_values.append(sum(train_y[sample[n:-1]]))


print([train_y[sample[n:-1]][i] for i in range(n-1)])
plt.scatter(mse,[1,2,3,4,5,6,7,8])
plt.show()

