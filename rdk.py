from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate
import numpy as np
import gzip
import time
import pandas as pd
import pickle

trainData = pd.read_csv('train.csv.gz',compression="gzip")
train_labels = trainData.loc[:,"smiles"]

# Morgan fingerprint features 

trunk = 100000
bits = 32
features = np.zeros((bits,trunk))

for i,label in enumerate(train_labels):
	if i == trunk:
		break 
	if i % 10000 == 0:
		print "{0}0000 molecules done!".format(i / 10000)
	m = Chem.MolFromSmiles(label)
	fp1 = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=bits)
	features[:,i] = np.array(fp1)

features = features[:, features.any(axis=0)]
print features.shape

with gzip.open('pickled_features.tar.gz', 'wb') as out:
	pickle.dump(features, out)


# pd.csv_write(compression="gzip") 

