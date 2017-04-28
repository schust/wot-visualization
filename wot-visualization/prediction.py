import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/training2.csv', delimiter=',')

print(data.shape)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
print(data.shape)

# print(np.all(np.isfinite(X_train)))
# print(np.any(np.isnan(X_train)))
# print(data.columns)
# print(data.dtypes)

y = data['label']
print(y.shape)

X = data.drop('label', axis=1)

# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(X)
# X = pd.DataFrame(np_scaled)

# sel = VarianceThreshold()
# X = sel.fit_transform(X)

# X = SelectKBest(f_regression, k=10).fit_transform(X, y)
# X = pd.DataFrame(X)

# pca = PCA()
# pca.fit(X)
# X = pca.transform(X)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# for row in X_train.iterrows():
#     print(row)

# KNN
# for i in range(5, 2000, 5):
#     classifier = KNeighborsClassifier(n_neighbors=i)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#
#     acc = accuracy_score(y_test, y_pred)
#     print(str(i) + ': ' + str(acc))

# SVC
# classifier = SVC()

# GRADIENT BOOSTING
# classifier = GradientBoostingClassifier()

# ADABOOST
classifier = AdaBoostClassifier()

# RANDOM FOREST
# for i in range(5, 2000, 5):
#     classifier = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=5)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#
#     acc = accuracy_score(y_test, y_pred)
#     print(str(i) + ': ' + str(acc))

# NEURAL NET
# classifier = MLPClassifier(verbose=True, solver='sgd')

# NAIVE BAYES
# classifier = GaussianNB()

# MULTINOMIAL NAIVE BAYES
# classifier = MultinomialNB(fit_prior=False)

# BERNOULLI NB
# classifier = BernoulliNB(binarize=0.5)

# TREE
# classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('acc: ' + str(acc))

conf = confusion_matrix(y_test, y_pred)
print(conf)
