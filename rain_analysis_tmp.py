import sys
from pyspark.sql import SparkSession, functions, types, Row

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

spark = SparkSession.builder.appName('Analyze Reddit').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

weather_schema = types.StructType([
    types.StructField('station_id', types.StringType()),
    types.StructField('date', types.IntegerType()),
    types.StructField('element', types.StringType()),
    types.StructField('data_value', types.IntegerType()),
    types.StructField('obs_time', types.IntegerType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('city', types.StringType()),
])

comments_schema = types.StructType([
    types.StructField('subreddit', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('sentiment', types.StringType()),
])



    weather_path = 'weather-output/weather-'+city+'/part*'

    reddit_data = spark.read.json(reddit_path, schema=comments_schema)
    weather_data = spark.read.json(weather_path, schema=weather_schema)

    analyze_sentiment = functions.udf(get_sentiment, returnType=types.StringType())
    reddit_data = reddit_data.withColumn('sentiment', analyze_sentiment(reddit_data['body']))

    rainy_days = weather_data.filter((weather_data['element'] == "PRCP"))
    reddit_count = reddit_data.groupBy(['year','month','day','sentiment']).count()
    combined = reddit_count.join(rainy_days, ['year','month','day']).drop(rainy_days['obs_time'],rainy_days['date']).orderBy(['year','month','day','sentiment'])

    combined_pd = combined.toPandas()
    combined_pd['has_rain'] = np.where(combined_pd['data_value']>0, True, False)






if __name__=='__main__':
    city = sys.argv[1]
    # out_directory = sys.argv[2]
    main(city)