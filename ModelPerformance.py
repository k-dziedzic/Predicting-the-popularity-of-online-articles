from pyspark.sql import SparkSession
from sklearn.cross_validation import train_test_split
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

# split dataset to 70% training and 30% testing
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