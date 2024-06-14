import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from MultiBoostClassifier import MultiBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from mlxtend.evaluate import paired_ttest_5x2cv
from scipy.stats import levene


# Read Grade data
gradeData = pd.read_csv('Expanded_data_with_more_features.csv')
gradeData = gradeData.dropna() #19243 x 15 dataframe

label = LabelEncoder()

# Converts strings to numerical values
gradeData = gradeData.apply(label.fit_transform)
gradeData = gradeData.drop(gradeData.columns[0], axis=1)

gradeData['MathScore'] = np.floor(gradeData['MathScore'] / 10)
gradeData['MathScore'] = gradeData['MathScore'].replace(10, 9)
gradeData['ReadingScore'] = np.floor(gradeData['ReadingScore'] / 10)
gradeData['ReadingScore'] = gradeData['ReadingScore'].replace(10, 9)
gradeData['WritingScore'] = np.floor(gradeData['WritingScore'] / 10)
gradeData['WritingScore'] = gradeData['WritingScore'].replace(10, 9)

X = gradeData.drop('MathScore', axis=1).values
y = gradeData['MathScore'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Fit models
model = MultiBoostClassifier(50, baseLearner = DecisionTreeClassifier(max_depth=5)).fit(X_train, y_train)
y_pred = model.predict(X_test)

model2 = AdaBoostClassifier(algorithm='SAMME').fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

model3 = RandomForestClassifier().fit(X_train, y_train)
y_pred3 = model3.predict(X_test)

model4 = BaggingClassifier().fit(X_train, y_train)
y_pred4 = model4.predict(X_test)

# Cross-validation scores
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

scores2 = cross_val_score(model2, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))

scores3 = cross_val_score(model3, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores3.mean(), scores3.std()))

scores4 = cross_val_score(model4, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.3f" % (scores4.mean(), scores4.std()))

# Model accuracy for different values of T
TList = [1, 10, 20, 30, 40, 50, 60, 70, 80 ,90, 100]
TResult = np.zeros((4, len(TList)))

for i in TList:
    model = MultiBoostClassifier(T = i, baseLearner = DecisionTreeClassifier(max_depth=5)).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    TResult[0, int(i/10)] = accuracy_score(y_test, y_pred)
    
    model2 = AdaBoostClassifier(n_estimators=i, algorithm='SAMME').fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    TResult[1, int(i/10)] = accuracy_score(y_test, y_pred2)
    
    model3 = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    TResult[2, int(i/10)] = accuracy_score(y_test, y_pred3)
    
    model4 = BaggingClassifier(n_estimators=i).fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    TResult[3, int(i/10)] = accuracy_score(y_test, y_pred4)

print(TResult)

# 5x2 cross-validation t-test
t, p = paired_ttest_5x2cv(model, model2, X, y, scoring='accuracy', random_seed=123)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t2, p2 = paired_ttest_5x2cv(model, model3, X, y, scoring='accuracy', random_seed=123)

print('t statistic: %.3f' % t2)
print('p value: %.3f' % p2)

t3, p3 = paired_ttest_5x2cv(model, model4, X, y, scoring='accuracy', random_seed=123)

print('t statistic: %.3f' % t3)
print('p value: %.3f' % p3)

paired_ttest_5x2cv()

# ROC Curve Generating
# Adapted from DataCamp Article on Learning Curves (Pykes, 2022) and sklearn Documentaton
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model4,
    X=X,
    y=y,
    cv=5,
    scoring='accuracy',
    train_sizes = [0.1, 0.33, 0.55, 0.78, 1]
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
plt.plot(train_sizes, train_mean, label="Training Set")
plt.plot(train_sizes, test_mean, label="Testing Set")

plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")

plt.show()

# Levene Test
stat1, p1 = levene(scores, scores2)
stat2, p2 = levene(scores, scores3)
stat3, p3 = levene(scores, scores4)

print('Levene statistic: %.3f' % stat1)
print('p value: %.3f' % p1)

print('Levene statistic: %.3f' % stat2)
print('p value: %.3f' % p2)

print('Levene statistic: %.3f' % stat3)
print('p value: %.3f' % p3)