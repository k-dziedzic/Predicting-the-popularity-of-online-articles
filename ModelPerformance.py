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


# ANALYSIS

list1=[]
list2=[]
list3=[]

list4=[]
list5=[]

list6=[]
list7=[]

list8=[]
list9=[]

for i in range(0,100,5):
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.049+i/100.0, random_state=0)

    # Linear Regression
    t0 = time()
    lr = linear_model.LinearRegression()
    clf_lr = lr.fit(X_test1, y_test1)
    accuracy_lr = clf_lr.score(X_test, y_test)
    t1 = time()
    timeElapsed_lr=t1-t0
    list1.append(i)
    list2.append(accuracy_lr)
    list3.append(timeElapsed_lr)

    # Random Forest
    t2 = time()
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf_rf = rf.fit(X_test1, y_test1)
    accuracy_rf = clf_rf.score(X_test, y_test)
    t3 = time()
    timeElapsed_rf = t3 - t2
    list4.append(accuracy_rf)
    list5.append(timeElapsed_rf)

    # KNN
    t4 = time()
    knn = KNeighborsClassifier()
    clf_knn = knn.fit(X_test1, y_test1)
    accuracy_knn = clf_knn.score(X_test, y_test)
    t5 = time()
    timeElapsed_knn = t5 - t4
    list6.append(accuracy_knn)
    list7.append(timeElapsed_knn)

    # Naive Bayes
    t6 = time()
    nb = BernoulliNB()
    clf_nb = nb.fit(X_test1, y_test1)
    accuracy_nb = clf_nb.score(X_test, y_test)
    t7 = time()
    timeElapsed_nb = t7 - t6
    list8.append(accuracy_nb)
    list9.append(timeElapsed_nb)


# view reliability diagram
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot('211')
plt.plot(list1, list2, label="Linear Regression")
leg = plt.legend()
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
plt.title('Accuracy with change train data')
plt.xlabel("% test size")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot('212')
plt.subplots_adjust(hspace=0.3)
plt.plot(list1, list4, label="Random Forrest")
plt.plot(list1, list6, label="KNN")
plt.plot(list1, list8, label="Naive Bayes")
leg = plt.legend()
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
plt.title('Accuracy with change train data')
plt.xlabel("% test size")
plt.ylabel("Accuracy")
plt.grid(True)



plt.show()


