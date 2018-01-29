from pyspark.ml.linalg import DenseVector
from pyspark.shell import spark, sqlContext

# download data OnlineNewsPopularity
import pyensae

pyensae.download_data("OnlineNewsPopularity.zip",
                      url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("OnlineNewsPopularity").getOrCreate()

data = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load(
    "OnlineNewsPopularity/OnlineNewsPopularity.csv")

# Standardization

# Define the `input_data`
input_data = data.rdd.map(lambda x: (x[0], DenseVector(x[1:5])))

# Replace `data` with the new DataFrame
data = spark.createDataFrame(input_data, ["label", "features"])

# data.show()

# Import `StandardScaler`
from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(data)

# Transform the data in `df` with the scaler
scaled_data = scaler.transform(data)

# Inspect the result
print(scaled_data.take(1))
