from pyspark.ml.linalg import DenseVector
from pyspark.shell import spark, sqlContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pyensae

# download data OnlineNewsPopularity
pyensae.download_data("OnlineNewsPopularity.zip",
                      url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("OnlineNewsPopularity").getOrCreate()

    data = spark.read \
        .options(header="true", inferSchema="true") \
        .csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")

    print("Total number of rows: %d" % data.count())
    
    def convertColumn(df, names, newType):
        for name in names:
            df = df.withColumn(name, df[name].cast(newType))
        return df

    # Assign all column names to `columns`
    columns = [' timedelta', ' n_tokens_title', ' n_tokens_content', ' n_unique_tokens', ' n_non_stop_words',
               ' n_non_stop_unique_tokens', ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos',
               ' average_token_length', ' num_keywords', ' data_channel_is_lifestyle', ' data_channel_is_entertainment',
               ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_tech', ' data_channel_is_world',
               ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg',
               ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares', ' self_reference_max_shares',
               ' self_reference_avg_sharess', ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
               ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday', ' weekday_is_sunday',
               ' is_weekend', ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity',
               ' global_sentiment_polarity', ' global_rate_positive_words', ' global_rate_negative_words',
               ' rate_positive_words', ' rate_negative_words', ' avg_positive_polarity', ' min_positive_polarity',
               ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity', ' max_negative_polarity',
               ' title_subjectivity', ' title_sentiment_polarity', ' abs_title_subjectivity',
               ' abs_title_sentiment_polarity', ' shares']

    # Conver the `df` columns to `FloatType()`
    data = convertColumn(data, columns, FloatType())

    data = data.select(' shares', ' num_keywords', ' data_channel_is_lifestyle', ' data_channel_is_entertainment',
               ' data_channel_is_bus', ' data_channel_is_socmed', ' data_channel_is_tech', ' data_channel_is_world',
               ' kw_min_min', ' kw_max_min')

    # Define the `input_data`
    input_data = data.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

    # Replace `data` with the new DataFrame
    data = spark.createDataFrame(input_data, ["label", "features"])

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only

spark.stop
