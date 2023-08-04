import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.functions import when
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

spark = SparkSession.builder.appName('Analyze Reddit').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

weather_schema = types.StructType([
    types.StructField('station_id', types.StringType()),
    types.StructField('date', types.IntegerType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('city', types.StringType()),
    types.StructField('PRCP', types.FloatType()),
    types.StructField('SNOW', types.FloatType()),
    types.StructField('SNWD', types.FloatType()),
    types.StructField('TMAX', types.FloatType()),
    types.StructField('TMIN', types.FloatType()),
    types.StructField('TAVG', types.FloatType()),
    types.StructField('T_label', types.FloatType()),
])

comments_schema = types.StructType([
    types.StructField('subreddit', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('sentiment', types.StringType()),
])

sentimentAnalyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    sentiment = sentimentAnalyzer.polarity_scores(text)
    return sentiment_classification(sentiment['compound'])

def sentiment_classification(score):
    if score >= 0.05:
        return "positive"
    elif score > -0.05 and score < 0.05:
        return "neutral"
    else:
        return "negative"

def main(city):
    reddit_path = 'subreddit-output/subreddit-'+city+'/part*'
    weather_path = 'weather-output/weather-'+city+'/part*'
    # print(reddit_path, weather_path)

    reddit_data = spark.read.json(reddit_path, schema=comments_schema)
    weather_data = spark.read.json(weather_path, schema=weather_schema)

    analyze_sentiment = functions.udf(get_sentiment, returnType=types.StringType())
    reddit_data = reddit_data.withColumn('sentiment', analyze_sentiment(reddit_data['body'])).drop(reddit_data['day'])
    reddit_data.cache()
    # reddit_data.show()

    reddit_count = reddit_data.groupBy(['year','month','sentiment']).count()
    # reddit_count.show()

    rainy_days = weather_data.filter((weather_data['PRCP'].isNotNull()))
    rainy_days = rainy_days.withColumn('PRCP(mm)', weather_data['PRCP']/10)
    rainy_days = rainy_days.drop(rainy_days['date'], rainy_days['SNOW'], rainy_days['SNWD'], rainy_days['TMAX'], rainy_days['TMIN'], rainy_days['TAVG'], rainy_days['T_label'])
    
    
    rainy_days = rainy_days.groupBy(['year','month']).sum()
    rainy_days = rainy_days.drop('sum(year)','sum(month)', 'sum(day)', 'sum(PRCP)')
    rainy_days.cache()

    rainy_days.show()

    combined = reddit_count.join(rainy_days, ['year','month']).orderBy(['year','month','sentiment'])
    combined = combined.withColumn('has_rain',when(combined['sum(PRCP(mm))'] > 1, True).otherwise(False))
    combined.cache()

    combined.show()



    # ### PANDAS DATA FRAME ###
    combined_pd = combined.toPandas()
    # print(combined_pd)

    neg_combined_pd = combined_pd.loc[combined_pd['sentiment']=='negative']
    pos_combined_pd = combined_pd.loc[combined_pd['sentiment']=='positive']
    neutral_combined_pd = combined_pd.loc[combined_pd['sentiment']=='neutral']


    # neg_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='negative') & (combined_pd['has_rain']==True)]
    # pos_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='positive') & (combined_pd['has_rain']==True)]
    # neutral_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='neutral') & (combined_pd['has_rain']==True)]

    # neg_no_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='negative') & (combined_pd['has_rain']==False)]
    # pos_no_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='positive') & (combined_pd['has_rain']==False)]
    # neutral_no_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='neutral') & (combined_pd['has_rain']==False)]

    
    # #### LINEAR REGRESSION PLOT & CALCULATIONS ####
    # plt.plot(pos_combined_pd['count'],pos_combined_pd['sum(PRCP(mm))'], 'b.')
    # plt.plot(neg_combined_pd['count'],neg_combined_pd['sum(PRCP(mm))'], 'r.')
    # plt.plot(neutral_combined_pd['count'],neutral_combined_pd['sum(PRCP(mm))'], 'g.')
    # plt.legend(['positive','negative','neutral'])
    # plt.savefig(city+'-monthly-scatter.png')
    # # plt.show()

    pos_regression = stats.linregress(pos_combined_pd['count'],pos_combined_pd['sum(PRCP(mm))'])
    neg_regression = stats.linregress(neg_combined_pd['count'],neg_combined_pd['sum(PRCP(mm))'])
    neut_regression = stats.linregress(neutral_combined_pd['count'],neutral_combined_pd['sum(PRCP(mm))'])

    print("LINEAR REGRESSION: Positive p-value:", pos_regression.pvalue, ", r-value:",pos_regression.rvalue)
    print("LINEAR REGRESSION: Negative lin reg p-value:", neg_regression.pvalue, ", r-value:",neg_regression.rvalue)
    print("LINEAR REGRESSION: Neutral lin reg p-value:", neut_regression.pvalue, ", r-value:",neut_regression.rvalue)

    x = np.array(pos_combined_pd['sum(PRCP(mm))'])
    y = pos_combined_pd['count']
    X = np.stack([x], axis=1)

    plt.figure(1,figsize=(10,5))
    
    model = LinearRegression(fit_intercept = True)
    model.fit(X, y)

    plt.subplot(1,2,1)
    plt.plot(x, y, 'b.')
    plt.plot(x, model.predict(X), 'r-')
    

    model_poly = make_pipeline(
        PolynomialFeatures(degree=11, include_bias=True),
        LinearRegression(fit_intercept=False)
    )

    model_poly.fit(X, y)

    plt.subplot(1,2,2)
    plt.plot(x, y, 'b.')
    plt.plot(x, model_poly.predict(X), 'r-')
    
    # plt.show()


    x = np.array(neg_combined_pd['sum(PRCP(mm))'])
    y = neg_combined_pd['count']
    X = np.stack([x], axis=1)

    plt.figure(2,figsize=(10,5))
    
    model = LinearRegression(fit_intercept = True)
    model.fit(X, y)

    plt.subplot(1,2,1)
    plt.plot(x, y, 'c.')
    plt.plot(x, model.predict(X), 'r-')
    

    model_poly = make_pipeline(
        PolynomialFeatures(degree=11, include_bias=True),
        LinearRegression(fit_intercept=False)
    )

    model_poly.fit(X, y)

    plt.subplot(1,2,2)
    plt.plot(x, y, 'c.')
    plt.plot(x, model_poly.predict(X), 'r-')
    plt.savefig(city+'-monthly-scatter1.png')
    # plt.show()

    x = np.array(neutral_combined_pd['sum(PRCP(mm))'])
    y = neutral_combined_pd['count']
    X = np.stack([x], axis=1)

    plt.figure(3,figsize=(10,5))
    
    model = LinearRegression(fit_intercept = True)
    model.fit(X, y)

    plt.subplot(1,2,1)
    plt.plot(x, y, 'g.')
    plt.plot(x, model.predict(X), 'r-')
    

    model_poly = make_pipeline(
        PolynomialFeatures(degree=11, include_bias=True),
        LinearRegression(fit_intercept=False)
    )

    model_poly.fit(X, y)

    plt.subplot(1,2,2)
    plt.plot(x, y, 'g.')
    plt.plot(x, model_poly.predict(X), 'r-')

    plt.savefig(city+'-monthly-scatter2.png')
    # plt.show()



if __name__=='__main__':
    city = sys.argv[1]
    # out_directory = sys.argv[2]
    main(city)