#download data OnlineNewsPopularity
import pyensae
pyensae.download_data("OnlineNewsPopularity.zip", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

#import data from file .csv
import pandas
data = pandas.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")

#spliting data to train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2)

#print train  and test data
print("TRAIN DATA")
print (train)

print("\n\nTEST DATA")
print (test)

spark.stop