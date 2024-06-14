import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from MultiBoostClassifier import MultiBoostClassifier
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import paired_ttest_5x2cv
from scipy.stats import levene

#Read MNIST data
mnist_train = pd.read_csv('mnist_train.csv')
mnist_test = pd.read_csv('mnist_test.csv')

X_train = mnist_train.drop('label', axis=1).values
y_train = mnist_train['label'].values

X_test = mnist_test.drop('label', axis=1).values
y_test = mnist_test['label'].values

X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

#Fit models
model = MultiBoostClassifier(50, baseLearner = DecisionTreeClassifier(max_depth=5)).fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

model2 = AdaBoostClassifier(algorithm='SAMME').fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(accuracy_score(y_test, y_pred2))

model3 = RandomForestClassifier().fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print(accuracy_score(y_test, y_pred3))

model4 = BaggingClassifier().fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(accuracy_score(y_test, y_pred4))

#Cross-validation scores
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

scores2 = cross_val_score(model2, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))

scores3 = cross_val_score(model3, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores3.mean(), scores3.std()))

scores4 = cross_val_score(model4, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores4.mean(), scores4.std()))

#5x2 cross-validation t-test
t, p = paired_ttest_5x2cv(model, model2, X, y, scoring='accuracy', random_seed=123)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t2, p2 = paired_ttest_5x2cv(model, model3, X, y, scoring='accuracy', random_seed=123)

print('t statistic: %.3f' % t2)
print('p value: %.3f' % p2)

t3, p3 = paired_ttest_5x2cv(model, model4, X, y, scoring='accuracy', random_seed=123)

print('t statistic: %.3f' % t3)
print('p value: %.3f' % p3)

#Per Number Accuracy Vector
correctList = np.zeros(10)
amountList = np.zeros(10)

for x in range(0, len(y_test)):
    for i in range(10):
        if y_test[x] == i:
            amountList[i] += 1
        if y_test[x] == y_pred[x] and y_test[x] == i:
            correctList[i] += 1

propList = np.true_divide(correctList, amountList)

print(correctList)

print(propList)

#Levene's Test
stat1, p1 = levene(scores, scores2)
stat2, p2 = levene(scores, scores3)
stat3, p3 = levene(scores, scores4)

print('Levene statistic: %.3f' % stat1)
print('p value: %.3f' % p1)

print('Levene statistic: %.3f' % stat2)
print('p value: %.3f' % p2)

print('Levene statistic: %.3f' % stat3)
print('p value: %.3f' % p3)