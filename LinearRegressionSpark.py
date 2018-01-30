from pyspark.ml.linalg import DenseVector
from pyspark.shell import spark, sqlContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pyensae

# download data OnlineNewsPopularity
pyensae.download_data("OnlineNewsPopularity.zip",
                      url="https://archive.ics.uci.edu/ml/machine-learning-databases/00332/")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("OnlineNewsPopularity").getOrCreate()

    data = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load(
        "OnlineNewsPopularity/OnlineNewsPopularity.csv")


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
               ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday', ' weekday_is_sunday', ' is_weekend',
               ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity', ' global_sentiment_polarity',
               ' global_rate_positive_words', ' global_rate_negative_words', ' rate_positive_words', ' rate_negative_words',
               ' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity',
               ' min_negative_polarity', ' max_negative_polarity', ' title_subjectivity', ' title_sentiment_polarity',
               ' abs_title_subjectivity', ' abs_title_sentiment_polarity', ' shares']

    # Conver the `df` columns to `FloatType()`
    data = convertColumn(data, columns, FloatType())

    data = data.select(' shares',' n_non_stop_words')

    # standardization

    # Define the `input_data`
    input_data = data.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

    # Replace `data` with the new DataFrame
    data = spark.createDataFrame(input_data, ["label", "features"])

    # Initialize the `standardScaler`
    standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

    # Fit the DataFrame to the scaler
    scaler = standardScaler.fit(data)

    # Transform the data in `df` with the scaler
    scaled_data = scaler.transform(data)

    # Bulding a machine learning model
    # split the data into train and test sets
    train_data, test_data = scaled_data.randomSplit([.8, .2], seed=1234)

    # initialize `lr`
    lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # fit the data to the model
    linearModel = lr.fit(train_data)

    # generate predictions
    predicted = linearModel.transform(test_data)

    # extract the predictions and the "known" correct labels
    predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
    labels = predicted.select("label").rdd.map(lambda x: x[0])

    # zip `predictions` and `labels` into a list
    predictionAndLabel = predictions.zip(labels).collect()

    # zip `predictions` and `labels` into a list
    predictionAndLabel = predictions.zip(labels).collect()

    # coefficients for the model
    print("Coefficients")
    print(linearModel.coefficients)

    # intercept for the model
    print("\nIntercept")
    print(linearModel.intercept)

    # get the RMSE
    # how much error there is between two datasets comparing; smaller is better
    print("\nRMSE")
    print(linearModel.summary.rootMeanSquaredError)

    # get the R2
    # how close the data are to the fitted regression line; 0-the worst, 1-the best
    print("\nR2")
    print(linearModel.summary.r2)

    tab1=np.array(train_data.select('features_scaled').collect())
    tab2=np.array(train_data.select('label').collect())

    # view reliability diagram
    plt.figure(1)
    plt.subplot('221')
    plt.scatter(tab1,tab2)
    plt.title('Training data')
    plt.xlabel(train_data.columns[2])
    plt.ylabel(train_data.columns[0])
    plt.grid(True)

    tab1=np.array(test_data.select('features_scaled').collect())
    tab2=np.array(test_data.select('label').collect())

    plt.subplot('222')
    plt.scatter(tab1,tab2)
    plt.title('Testing data')
    plt.xlabel(test_data.columns[2])
    plt.ylabel(test_data.columns[0])
    plt.grid(True)
    #
    # tab1=np.array(predicted.select('features_scaled').collect())
    # tab2=np.array(predicted.select('label').collect())
    # tab3=np.array(predicted.select('prediction').collect())
    # tab1=tab1.reshape(7905,1)
    #
    # plt.subplot('223')
    # plt.scatter(tab1, tab2)
    # plt.plot(tab1 ,tab2, color='green', linewidth=3)
    # plt.title('Learned linear regression')
    # plt.xlabel(train_data.columns[2])
    # plt.ylabel(train_data.columns[0])
    # plt.grid()
    #
    # plt.subplot('223')
    # plt.scatter(tab3, tab4)
    # plt.plot(tab4, tab5, color='green', linewidth=3)
    # plt.title('Learned linear regression')
    # plt.xlabel('x3')
    # plt.ylabel('target')
    # plt.grid()

    plt.show()

spark.stop