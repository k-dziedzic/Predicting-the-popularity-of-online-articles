from pyspark import Row
from pyspark.shell import spark, sqlContext, sc

#download data OnlineNewsPopularity
import pyensae
pyensae.download_data("OnlineNewsPopularity.zip", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

from pyspark.sql import  SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("OnlineNewsPopularity").getOrCreate()

data = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load(
    "OnlineNewsPopularity/OnlineNewsPopularity.csv")

data=sc.textFile("OnlineNewsPopularity/OnlineNewsPopularity.csv").map(lambda line: line.split(", ")).filter(lambda line: len(line)>1)

data.collect()

print(data.first)
print("\n")
print(data.top(2))

spark.stop

