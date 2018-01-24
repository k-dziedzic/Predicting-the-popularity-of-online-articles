import pyensae
from pyspark.sql import SparkSession
from pyspark.shell import sqlContext


# download data OnlineNewsPopularity
pyensae.download_data("OnlineNewsPopularity.zip", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

# create Spark Session
if __name__ == "__main__":spark = SparkSession.builder.appName("OnlineNewsPopularity").getOrCreate()

# read data from .csv file
data = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load(
    "OnlineNewsPopularity/OnlineNewsPopularity.csv")

# show data table
data.show()

# show columns belongs to table
print(data.columns)

# show schema with format of column
data.printSchema()

# show  2x10 table with data from following columns
data.select(' shares',' n_tokens_title').show(10)

# show describe table
data.describe().show()