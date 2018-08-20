# GBDT on neighor(parameter anadjusted)

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


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
	gbm0 = GradientBoostingClassifier(random_state=10)
	gbm0.fit(X, y)
	z_pre = gbm0.predict(Z)
	z_pre_prob = gbm0.predict_proba(Z)[:, 1]
	print("Accuracy: %.4g" % metrics.accuracy_score(z, z_pre))
	print("Pro_Accuracy: %.4g" % metrics.roc_auc_score(z, z_pre_prob))


if __name__ == '__main__':
	main()
