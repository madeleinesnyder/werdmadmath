# file I/O
import gzip
import numpy as np

train = []
values = []
strings = []
with gzip.open('train.csv.gz') as f:
	for line in f:
		if line[0] == 's': ## pass over first line
			continue
		zzz = line.split(',')
		train.append([bool(x) for x in zzz[1:-1]])
		values.append(float(zzz[-1]))
		strings.append(zzz[0])

train_x = np.array(train,dtype=bool)
train_y = np.array(values)

# model building
import sklearn.ensemble as ens
clf = ens.RandomForestRegressor(n_estimators=75, n_jobs=-1, warm_start=True)
uhhh = clf.fit(train_x[1:1000], train_y[1:1000])
xhat = clf.predict(train_x[1000:2000])

print sum([(train_y[1000+i] - xhat[i])**2 for i in range(1000)])/1000.0


'''
# plot
import matplotlib.pyplot as plt 
plt.plot(range(1000), xhat, 'ro', range(1000), train_y[1000:2000], 'bo')
plt.show()



# predicting
test_values = []
with gzip.open('test.csv.gz') as g:
	for line in g:
		if line[0] == 's': ## pass over first line
			continue
		test_values.append([bool(x) for x in line.split(',')[1:-1]])

test_values = np.array(test_values, dtype=bool)'''