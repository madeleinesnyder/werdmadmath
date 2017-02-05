import numpy as np
import pandas as pd
import time
from sklearn import linear_model

start_time = time.time()

zero_idx = []
allZeros = []

trainData = pd.read_csv('../train.csv.gz',compression="gzip")
train_y = trainData.loc[:,"gap"]
train_x = trainData.loc[:,:]
train_x = train_x.drop("smiles",axis=1)
train_x = train_x.drop("gap",axis=1)
zero_idx = train_x.apply(lambda x: np.all(x==0))
for i, value in enumerate(zero_idx):
	if value == True:
		allZeros.append(i)
	if i ==True:
		print(zero_idx.iloc[i])
print(zero_idx.iloc)
print(zero_idx[i]==True for i in zero_idx)
print(train_x.iloc[zero_idx[i]==True])


print("--- %s seconds ---" % (time.time() - start_time))

# Make the k-fold thingy
k = 8

#1. Randomly choose 20000

