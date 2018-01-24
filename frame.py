from pyspark import Row, SparkContext
from pyspark.ml.linalg import DenseVector
from pyspark.shell import spark, sqlContext, sc

#download data OnlineNewsPopularity
import pyensae
pyensae.download_data("OnlineNewsPopularity.zip", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

from pyspark.sql import  SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("OnlineNewsPopularity").getOrCreate()

data = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load(
    "OnlineNewsPopularity/OnlineNewsPopularity.csv")

# data.show()

# print(data.columns)
# data.printSchema()
# data.select(' shares',' n_tokens_title').show(10)
# data.describe().show()

data=data.select(' shares', ' n_tokens_title', ' n_tokens_content', ' n_unique_tokens')

#standardization

# Define the `input_data`
input_data=data.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace `data` with the new DataFrame
data=spark.createDataFrame(input_data, ["label", "features"])

data.show()

spark.stop

