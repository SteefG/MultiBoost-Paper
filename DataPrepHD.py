import pandas as pd
import numpy as np
from MultiBoostClassifier import MultiBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from mlxtend.evaluate import paired_ttest_5x2cv
from scipy.stats import levene


# Read Heart Disease data
hdData = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
hdData = hdData.dropna() #1190 x 12 dataframe

X = hdData.drop('target', axis=1).values
y = hdData['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Fit models
model = MultiBoostClassifier(50, baseLearner = DecisionTreeClassifier(max_depth=1)).fit(X_train, y_train)

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


#ROC Curve generating code adpated from sklearn documentation

display = RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()

display2 = RocCurveDisplay.from_estimator(model2, X_test, y_test)
plt.show()

display3 = RocCurveDisplay.from_estimator(model3, X_test, y_test)
plt.show()

display4 = RocCurveDisplay.from_estimator(model4, X_test, y_test)
plt.show()

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

#Levene test for variance equality

stat1, p1 = levene(scores, scores2)
stat2, p2 = levene(scores, scores3)
stat3, p3 = levene(scores, scores4)

print('Levene statistic: %.3f' % stat1)
print('p value: %.3f' % p1)

print('Levene statistic: %.3f' % stat2)
print('p value: %.3f' % p2)

print('Levene statistic: %.3f' % stat3)
print('p value: %.3f' % p3)