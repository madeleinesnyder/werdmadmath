from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate
import numpy as np
import gzip
import time
import pandas as pd
import pickle

trainData = pd.read_csv('../test.csv.gz',compression="gzip")
train_labels = trainData.loc[:,"smiles"]

# Morgan fingerprint features 
bits = 256

features = np.zeros((len(train_labels), bits))
for i,label in enumerate(train_labels):
	if i % 10000 == 0:
		print "{0}0000 molecules done!".format(i / 10000 + 1)
	m = Chem.MolFromSmiles(label)
	fp1 = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=bits)
	features[i,:] = np.array(fp1, dtype=bool)

try:
	features = features[:, features.any(axis=0)]
except:
	features = features[features.any(axis=0), :]

pd.DataFrame(features).to_csv('../test_256features.csv.gz', compression='gzip')

