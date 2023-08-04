import sys
from pyspark.sql import SparkSession, functions, types, Row
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

def main(city):
    weather_path = 'weather-output/weather-'+city+'/part*'

    weather_data = spark.read.json(weather_path, schema=weather_schema)

    # test = weather_data.filter((weather_data['element'] == "PSUN") | (weather_data['element'] == "TSUN"))
    # test.show()
    rainy_days = weather_data.filter((weather_data['element'] == "PRCP")).select(['element','data_value']).groupBy('element').count()

    # elements = weather_data.select(['element','data_value']).groupBy('element').count()

    rainy_days.show()
   


if __name__=='__main__':
    city = sys.argv[1]
    # out_directory = sys.argv[2]
    main(city)