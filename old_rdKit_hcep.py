from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate
import numpy as np
import gzip
import numpy as np
import time
import pandas as pd

trainData = pd.read_csv('../train.csv.gz',compression="gzip")
train_labels = trainData.loc[:,"smiles"]
print(train_labels)


# Morgan fingerprint features 

features = np.zeros((1024,1000000))

for i,label in enumerate(train_labels):
	m = Chem.MolFromSmiles(label)
	fp1 = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=512)
	features[:,i] = np.array([fp1])


for i,feature in enumerate(features):
	if (sum(feature) == 0):
		print('all zeros in feature ' + str(i))

fp1 = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024)
fp1 = np.array([fp1])
print(fp1)

# info={}
# fp = AllChem.GetMorganFingerprint(m,2,bitInfo=info)
# print(info)

