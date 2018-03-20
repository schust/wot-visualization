import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print('Started at ' + str(datetime.datetime.now()))

data = pd.read_csv('~/Dropbox/replays/201803101025.csv', delimiter=',')

print(data.shape)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
data = data.loc[:, (data != data.iloc[0]).any()]
print(data.shape)

# print(np.all(np.isfinite(X_train)))
# print(np.any(np.isnan(X_train)))
# print(data.columns)
# print(data.dtypes)

y = data['label']
print(y.shape)

X = data.drop('label', axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled)

# sel = VarianceThreshold()
# X = sel.fit_transform(X)
#
# X = SelectKBest(f_regression, k=200).fit_transform(X, y)
# X = pd.DataFrame(X)

# pca = PCA()
# pca.fit(X)
# X = pca.transform(X)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# for row in X_train.iterrows():
#     print(row)

# KNN
print('KNN')
classifier = KNeighborsClassifier(n_neighbors=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# SVC
# 'linear' 0.7633
# 'poly' 0.4917
# 'rbf' 0.7512
# 'sigmoid' 0.7443
# print('SVC')
# classifier = SVC(verbose=True, kernel='linear')
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# GRADIENT BOOSTING
# print('Gradient Boosting')
# classifier = GradientBoostingClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# ADABOOST
# print('Adaboost')
# classifier = AdaBoostClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# RANDOM FOREST
# print('Random Forest')
# classifier = RandomForestClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# NEURAL NET
# print('MLP')
# classifier = MLPClassifier(verbose=True, solver='adam', max_iter=10000)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# NAIVE BAYES
# print("Naive Bayes")
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# MULTINOMIAL NAIVE BAYES
# print('Multinomial Naive Bayes')
# classifier = MultinomialNB(fit_prior=False)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# BERNOULLI NB
# print('Bernoulli Naive Bayes')
# classifier = BernoulliNB(binarize=0.1)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# TREE
# print('Simple Decision Tree')
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# LOGREG 'newton-cg', 'lbfgs', 'liblinear', 'sag'
# print('Logistic Regression')
# classifier = LogisticRegression(penalty='lbfgs')
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)


# SGD
# print('SVM with Stochastic Gradient Descent')
# classifier = SGDClassifier(penalty='l1')
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('acc: ' + str(acc))

conf = confusion_matrix(y_test, y_pred)
print(conf)




# classifier = KNeighborsClassifier(3)
# classifier = SVC(kernel="linear", C=0.025)  # 0.767352703793
# classifier = DecisionTreeClassifier(max_depth=5)  # 0.75464083938
# classifier = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=10)
# classifier = MLPClassifier(alpha=1)  # 0.758474576271
# classifier = AdaBoostClassifier()  # 0.760088781275
# classifier = GaussianNB()  # 0.604721549637

