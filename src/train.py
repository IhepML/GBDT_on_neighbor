# GBDT on neighor(parameter adjusted)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV


def main():
	# get the data
	train = pd.read_csv('../data/train.csv')
	test = pd.read_csv('../data/test.csv')
	x_columns = [x for x in train if x in ['r', 'phi', 'lt', 'rt', 'mdetmt0', 'le', 'me', 're']]
	X = train[x_columns]
	y = train['isSignal']
	z_columns = [x for x in test if x in ['r', 'phi', 'lt', 'rt', 'mdetmt0', 'le', 'me', 're']]
	Z = test[z_columns]
	z = test['isSignal']

	# train and test
	gbm0 = GradientBoostingClassifier(random_state=10, learning_rate=0.1, n_estimators=80, min_samples_leaf=20,
									  max_features='sqrt', subsample=0.8, max_depth=13, min_samples_split=500)
	gbm0.fit(X, y)
	z_pre = gbm0.predict(Z)
	z_pre_prob = gbm0.predict_proba(Z)[:, 1]
	print("Accuracy: %.4g" % metrics.accuracy_score(z, z_pre))
	print("Pro_Accuracy: %.4g" % metrics.roc_auc_score(z, z_pre_prob))

	# plot fpr and tpr
	fpr, tpr, thresholds = roc_curve(z, z_pre_prob)
	roc_auc = auc(fpr, tpr)
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlim([-0.1, 1.2])
	plt.ylim([-0.1, 1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()


if __name__ == '__main__':
	main()
