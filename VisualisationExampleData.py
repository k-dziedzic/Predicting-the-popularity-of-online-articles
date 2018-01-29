#download data OnlineNewsPopularity
import pyensae
pyensae.download_data("OnlineNewsPopularity.zip", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

#import data from file .csv
import pandas
data = pandas.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")

#list of the feature column's names
n_tokens_title=data.ix[:,2]
shares_column = data.ix[:,60]

#print column 'shares'
print("shares")
print(shares_column)
print("\n")

#print column 'feature'
print("n_tokens_title")
print(n_tokens_title)