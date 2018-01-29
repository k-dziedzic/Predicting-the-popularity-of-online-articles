from pyspark.sql import SparkSession
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from time import time
import pandas as pd
import pyensae

# download data OnlineNewsPopularity
pyensae.download_data("OnlineNewsPopularity.zip",
                      url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

# create Spark Session
if __name__ == "__main__":
    spark = SparkSession.builder.appName("OnlineNewsPopularity").getOrCreate()

# read .csv from provided dataset
csv_filename = "OnlineNewsPopularity/OnlineNewsPopularity.csv"
# df=pd.read_csv(csv_filename,index_col=0)
df = pd.read_csv(csv_filename)

popular = df[' shares'] >= 2000
unpopular = df[' shares'] < 2000
df.loc[popular, ' shares'] = 1
df.loc[unpopular, ' shares'] = 0

features = list(df.columns[2:60])

# split dataset to 60% training and 40% testing
X_train, X_test, y_train, y_test = train_test_split(df[features], df[' shares'], test_size=0.3, random_state=0)

# Linear Regression accuracy and time elapsed caculation
t0 = time()
print("Linear Regression")
lr = linear_model.LinearRegression()
clf_lr = lr.fit(X_train, y_train)
print("Acurracy: ", clf_lr.score(X_test, y_test))
t1 = time()
print("Time elapsed: ", t1 - t0)

# Cross validation for Linear Regression
tt0 = time()
print("Cross validation RESULT")
# Evaluate a score by cross-validation
scores = cross_val_score(lr, df[features], df[' shares'], cv=5)
print(scores)
print(scores.mean())
tt1 = time()
print("Time elapsed: ", tt1 - tt0)
print("\n")

# Random Forest accuracy and time elapsed caculation
t2 = time()
print("RandomForest")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf_rf = rf.fit(X_train, y_train)
print("Acurracy: ", clf_rf.score(X_test, y_test))
t3 = time()
print("Time elapsed: ", t3 - t2)

# Cross validation for Random Forrest
tt2 = time()
print("Cross validation RESULT")
scores = cross_val_score(rf, df[features], df[' shares'], cv=5)
print(scores)
print(scores.mean())
tt3 = time()
print("Time elapsed: ", tt3 - tt2)
print("\n")

# KNN accuracy and time elapsed caculation
t4 = time()
print("KNN")
# knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier()
clf_knn = knn.fit(X_train, y_train)
print("Acurracy: ", clf_knn.score(X_test, y_test))
t5 = time()
print("Time elapsed: ", t5 - t4)

# Cross validation for KNN
tt4 = time()
print("Cross validation RESULT")
scores = cross_val_score(knn, df[features], df[' shares'], cv=5)
print(scores)
print(scores.mean())
tt5 = time()
print("Time elapsed: ", tt5 - tt4)
print("\n")

# Naive Bayes accuracy and time elapsed caculation
t6 = time()
print("NaiveBayes")
nb = BernoulliNB()
clf_nb = nb.fit(X_train, y_train)
print("Acurracy: ", clf_nb.score(X_test, y_test))
t7 = time()
print("Time elapsed: ", t7 - t6)

# Cross-validation for Naive Bayes
tt6 = time()
print("Cross validation RESULT")
scores = cross_val_score(nb, df[features], df[' shares'], cv=5)
print(scores)
print(scores.mean())
tt7 = time()
print("Time elapsed: ", tt7 - tt6)
print("\n")

