import pickle, numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# file_dissimilarity_score.pkl contains a dictionary with key as video name and value as the sum of
# dissimilarity scores of all chunks of that video 
with open('file_dissimilarity_score.pkl', 'rb') as handle:
    test_dissimilarity_score = pickle.load(handle)

# file_target.pkl contains a dictionary with key as video name and value as the true target (real/fake)
# for that video
with open('file_target.pkl', 'rb') as handle:
    test_target = pickle.load(handle)

# file_number_of_chunks.pkl contains a dictionary with key as video name and value as the number of chunks 
# in that video
with open('file_number_of_chunks.pkl', 'rb') as handle:
    test_number_of_chunks = pickle.load(handle)


with open('train_dissimilarity_score.pkl', 'rb') as handle:
    train_dissimilarity_score = pickle.load(handle)
with open('train_target.pkl', 'rb') as handle:
    train_target = pickle.load(handle)
with open('train_number_ofile_number_of_chunks.pkl', 'rb') as handle:
    train_number_of_chunks = pickle.load(handle)

Real_mean_dissimilarity_score = {}
Fake_mean_dissimilarity_score = {}
count1, count2 = 0, 0

for video, score in train_dissimilarity_score.items():
	tar = train_target[video]
	score = train_dissimilarity_score[video]
	num_chunks = train_number_of_chunks[video]
	mean_dissimilarity_score = (score.item()) / num_chunks
	print('name: ' + str(video) + ',tar: ' + str(tar) + ',mean_dissimilarity_score:' + str(
		mean_dissimilarity_score))
	if tar == 0:
		Fake_mean_dissimilarity_score[video] = mean_dissimilarity_score
		count1 += 1
	else:
		Real_mean_dissimilarity_score[video] = mean_dissimilarity_score
		count2 += 1

max_fake = max(Fake_mean_dissimilarity_score.values())
min_fake = min(Fake_mean_dissimilarity_score.values())
max_real = max(Real_mean_dissimilarity_score.values())
min_real = min(Real_mean_dissimilarity_score.values())

print('count1: ' + str(count1) + ', count2: ' + str(count2))
print('fake mds:',min_fake,max_fake)
print('real mds:',min_real, max_real )

if max_real <= min_fake:
	threshold = (max_real + min_fake)/2
else:
	tmp = (max_real - min_fake)/20
	highest_auc = 0
	best_thred = min_fake
	for i in range(20):
		thred = min_fake + tmp*i
		y_tar = np.zeros((len(train_target), 1))
		y_pred = np.zeros((len(train_target),1))
		count = 0
		for video,score in train_dissimilarity_score.items():
			tar = train_target[video]
			score = train_dissimilarity_score[video]
			num_chunks = train_number_of_chunks[video]
			mean_dissimilarity_score = (score.item()) / num_chunks
			#print('mean_dissimilarity_score:', mean_dissimilarity_score)
			if mean_dissimilarity_score >= thred:
				# predicted target is fake
				pred = 0
			else:
				# predicted target is real
				pred = 1

			y_tar[count,0] = tar
			y_pred[count,0] = pred
			count += 1
			#print('num ' + str(count) + ':' + video+ '  ' + str(tar) + '  ' + str(pred))
		#print('Frame wise AUC is: '+str(roc_auc_score(y_tar, y_pred)))
		auc = roc_auc_score(y_tar, y_pred)
		print('thred: ' + str(thred) + ' auc: ' + str(auc))
		if auc > highest_auc:
			highest_auc = auc
			best_thred = thred

	threshold = best_thred

print('threshold is: '+ str(threshold))



y_tar = np.zeros((len(test_target),1))
y_pred = np.zeros((len(test_target),1))
count = 0
for video,score in test_dissimilarity_score.items():
	tar = test_target[video]
	score = test_dissimilarity_score[video]
	num_chunks = test_number_of_chunks[video]
	mean_dissimilarity_score = (score.item()) / num_chunks
	#print('mean_dissimilarity_score:', mean_dissimilarity_score)
	if mean_dissimilarity_score >= threshold:
		# predicted target is fake
		pred = 0
	else:
		# predicted target is real
		pred = 1

	y_tar[count,0] = tar
	y_pred[count,0] = pred
	count += 1
	#print('num ' + str(count) + ':' + video+ '  ' + str(tar) + '  ' + str(pred))
print('Frame wise AUC is: '+str(roc_auc_score(y_tar, y_pred)))
print(confusion_matrix(y_tar, y_pred))