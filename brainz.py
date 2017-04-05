import gzip
import numpy as np
from operator import itemgetter
import csv
from itertools import islice
import musicbrainzngs

musicbrainzngs.set_useragent('python', '2.7', contact='madeleinesnyder@college.harvard.edu')

# Load the training data.
keys = []
with open('artists.csv', 'r') as artists:
    artists_csv = csv.reader(artists, delimiter=',', quotechar='"')
    next(artists_csv, None) # skips the header
    for row in artists_csv:
        artist_key   = row[0]
        artist_name = row[1]
        keys.append(artist_key)
    
    artist_info = musicbrainzngs.get_artist_by_id(artist_key)


artist_info_dict = {}
for i in xrange(100):

	temp_list = musicbrainzngs.get_artist_by_id(keys[i])['artist']
	if 'life-span' in temp_list:
		artist_info_dict[keys[i]] = [musicbrainzngs.get_artist_by_id(keys[i])['artist']['life-span']['begin']]
	# elif 'area' in temp_list:
	# 	artist_info_dict[keys[i]].append(musicbrainzngs.get_artist_by_id(keys[i])['artist']['area']['name'])
	else:
		artist_info_dict[keys[i]] = 'nan'


	# artist id = [country, band inception date, band end date, age of band (inception-end),type (group or person)]
	# insert a nan if there is a KeyError (HOW?)
	#artist_info_dict[keys[i]] = [musicbrainzngs.get_artist_by_id(keys[i])['artist']['area']['name']] #musicbrainzngs.get_artist_by_id(keys[i])['artist']['life-span']['begin'],musicbrainzngs.get_artist_by_id(keys[13])['artist']['life-span']['end'],musicbrainzngs.get_artist_by_id(keys[13])['artist']['type']]
	# if KeyError:
	# 	artist_info_dict[keys[i]] = 'nan'

for key, value in artist_info_dict.iteritems() :
    print value

# Make matrix of training data
train_data = []
with open('train.csv', 'r') as train:
	training_csv = csv.reader(train, delimiter=',', quotechar='"')
	next(training_csv, None) # skips the header
	for row in training_csv:
		train_data.append(row)

print artist_info_dict

for i in xrange(2000):
	if keys[i] in artist_info_dict:
		print artist_info_dict[keys[i]]
	else:
		print("not in dict")

# Build a log-linear Poisson model 
import statsmodels.api as sm

# train_data[:,1] is the artist key so want: artist_info_dict[train_data[1][1]]
train_data.exog = sm.add_constant(train_data.exog)

# Instantiate a gamma family model with the default link function.
pois_model = sm.GLM(train_data.endog, train_data.exog, family=sm.families.Poisson(Log))
pois_results = pois_model.fit()

