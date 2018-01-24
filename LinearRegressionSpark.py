#download data OnlineNewsPopularity
import pyensae
pyensae.download_data("OnlineNewsPopularity.zip", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

#import data from file .csv
import pandas
data = pandas.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")

#list of the feature column's names
dependent_column = data.ix[:,1]
shares_column = data.ix[:,60]

#spliting data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dependent_column,shares_column, test_size=0.01)

#print train  and test data
print("TRAIN DATA")
print (X_train)

print("TRAIN DATA")
print (X_test)

import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(X_train,Y_train)
plt.title('Training data')
plt.grid(True)

plt.show()

