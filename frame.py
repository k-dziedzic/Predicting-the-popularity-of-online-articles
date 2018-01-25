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

from pyspark.sql.types import *

def convertColumn(df, names, newType):
  for name in names:
     df = df.withColumn(name, df[name].cast(newType))
  return df

# Assign all column names to `columns`
columns = [' timedelta', ' n_tokens_title', ' n_tokens_content', ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos', ' average_token_length', ' num_keywords', ' data_channel_is_lifestyle', ' data_channel_is_entertainment', ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_tech', ' data_channel_is_world', ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg', ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares', ' self_reference_max_shares', ' self_reference_avg_sharess', ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday', ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday', ' weekday_is_sunday', ' is_weekend', ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity', ' global_sentiment_polarity', ' global_rate_positive_words', ' global_rate_negative_words', ' rate_positive_words', ' rate_negative_words', ' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity', ' max_negative_polarity', ' title_subjectivity', ' title_sentiment_polarity', ' abs_title_subjectivity', ' abs_title_sentiment_polarity', ' shares']

# print(data.printSchema)

# Conver the `df` columns to `FloatType()`
data = convertColumn(data, columns, FloatType())

data = data.select(' shares', ' timedelta', ' n_tokens_title', ' n_tokens_content', ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens')

#standardization

# Define the `input_data`
input_data=data.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace `data` with the new DataFrame
data=spark.createDataFrame(input_data, ["label", "features"])

# data.show()

# Import `StandardScaler`
from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler=StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler=standardScaler.fit(data)

# Transform the data in `df` with the scaler
scaled_data=scaler.transform(data)

# Inspect the result
# print(scaled_data.take(2))

# Bulding a machine learning model
# split the data into train and test sets
train_data, test_data = scaled_data.randomSplit([.8,.2],seed=1234)

print("TRAIN DATA")
print(train_data.show(10))
# print(train_data[['features_scaled']].show())
print("TEST DATA")
print(test_data.show(10))

spark.stop

