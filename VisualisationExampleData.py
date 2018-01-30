
#download data OnlineNewsPopularity
import pyensae
pyensae.download_data("OnlineNewsPopularity.zip", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

#import data from file .csv
import pandas
data = pandas.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")

#list of the feature column's names
n_tokens_title=data.ix[:,2]
n_tokens_content = data.ix[:,3]
num_keywords= data.ix[:,12]
num_hrefs=data.ix[:,7]
shares_column = data.ix[:,60]

# #print column 'shares'
# print("shares")
# print(shares_column)
# print("\n")
#
# #print column 'feature'
# print("n_tokens_title")
# print(n_tokens_title)

#view reliability diagram
import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(n_tokens_title,shares_column)
plt.title('Visualisation')
plt.xlabel("n_tokens_title")
plt.ylabel("shares")
plt.grid(True)

plt.savefig('chart/n_tokens_title_shares_column.png')

plt.figure(2)
plt.scatter(n_tokens_content,shares_column)
plt.title('Visualisation')
plt.xlabel("n_tokens_content")
plt.ylabel("shares")
plt.grid(True)

plt.savefig('chart/n_tokens_content_shares_column.png')

plt.figure(3)
plt.scatter(num_hrefs,shares_column)
plt.title('Visualisation')
plt.xlabel("num_hrefs")
plt.ylabel("shares")
plt.grid(True)

plt.savefig('chart/num_hrefs_shares_column.png')

plt.figure(4)
plt.scatter(n_tokens_title,shares_column)
plt.scatter(num_keywords,shares_column)
plt.title('Visualisation')
plt.xlabel("n_tokens_title/num_keywords")
plt.ylabel("shares")
plt.grid(True)

plt.savefig('chart/n_tokens_title_shares_column.png')

plt.show()