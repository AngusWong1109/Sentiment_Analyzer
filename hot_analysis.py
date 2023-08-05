import sys
from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import col, count, to_date, dayofmonth, month, year, when

spark = SparkSession.builder.appName('hot_analysis').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

weather_filepath = 'weather_output/weather-*'
subreddit_filepath = 'subreddit-output/subreddit-*'

weather_sample_filepath = 'sample_dataset/weather_output/weather-*'
subreddit_sample_filepath = 'sample_dataset/reddit_output/part-*'

weather_schema = types.StructType([
    types.StructField('station_id', types.StringType()),
    types.StructField('date', types.DateType()),
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
    types.StructField('T_label', types.StringType())
])

comments_schema = types.StructType([
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('body', types.StringType()),
    types.StructField('sentiment', types.StringType())
])

cities_to_work_on = []

def main():
    #read file
    if argv[1] == 'whole':
        weather = spark.read.csv(weather_filepath, schema=weather_schema)
        reddit = spark.read.json(subreddit_filepath, schema=comments_schema)
    else:
        weather = spark.read.csv(weather_sample_filepath, schema=weather_schema)
        reddit = spark.read.json(subreddit_sample_filepath, schema=comments_schema)
    weather = weather.sort('date', ascending=True)
    
    joined_data = weather.join(reddit, [weather.year == reddit.year, 
                                        weather.month == reddit.month,
                                        weather.day == reddit.day,
                                        weather.city == reddit.subreddit], how='left')
    joined_data = joined_data.select(
        'station_id', 'date', weather.year, weather.month, weather.day, 'city', 'TMAX', 'TMIN', 'TAVG', 'T_label', 'subreddit', 'body', 'sentiment'
    )
    
    joined_data = joined_data.sort(['year', 'month', 'day'], ascending=[True, True, True])
    joined_data = joined_data.dropna()
    joined_data = joined_data.cache()

    cold_weather = joined_data.filter(
        joined_data['T_label'] == 'cold'
    )
    hot_weather = joined_data.filter(
        joined_data['T_label'] == 'hot'
    )
    positive_post = joined_data.filter(
        joined_data['sentiment'] == 'positive'
    )
    negative_post = joined_data.filter(
        joined_data['sentiment'] == 'negative'
    )
    neutral_post = joined_data.filter(
        joined_data['sentiment'] == 'neutral'
    )
    
    model = make_pipeline(
        PolynomialFeatures(degree=4, include_bias=True),
        LinearRegression(fit_intercept=False)
    )
        
    #polynomial regression: hot and positive
    grouped_hot_pos = hot_weather.filter(hot_weather['sentiment'] == 'positive').groupBy(['subreddit', 'TAVG']).agg(count('sentiment').alias('num_of_pos'))
    pd_hot_pos = grouped_hot_pos.toPandas()
    pd_hot_pos = pd_hot_pos.astype({'TAVG':'float', 'num_of_pos':'int'})
    x = np.array(pd_hot_pos['TAVG'])
    y = pd_hot_pos['num_of_pos']
    X = np.stack([x], axis=1)
    model.fit(X, y)
    pd_hot_pos['prediction'] = model.predict(X)
    #linear regression: hot and positive
    reg_hot_pos = stats.linregress(pd_hot_pos['TAVG'], y)
    pd_hot_pos['lin_predict'] = reg_hot_pos.slope * pd_hot_pos['TAVG'] + reg_hot_pos.intercept
    
    #polynomial regression: hot and negative
    grouped_hot_neg = hot_weather.filter(hot_weather['sentiment'] == 'negative').groupBy(['subreddit', 'TAVG']).agg(count('sentiment').alias('num_of_neg'))
    pd_hot_neg = grouped_hot_neg.toPandas()
    pd_hot_neg = pd_hot_neg.astype({'TAVG':'float', 'num_of_neg':'int'})
    x = np.array(pd_hot_neg['TAVG'])
    y = pd_hot_neg['num_of_neg']
    X = np.stack([x], axis=1)
    model.fit(X, y)
    pd_hot_neg['prediction'] = model.predict(X)
    #linear regression: hot and positive
    reg_hot_neg = stats.linregress(pd_hot_neg['TAVG'], y)
    pd_hot_neg['lin_predict'] = reg_hot_neg.slope * pd_hot_neg['TAVG'] + reg_hot_neg.intercept
    
    #polynomial regression: hot and neural
    grouped_hot_neg = hot_weather.filter(hot_weather['sentiment'] == 'neutral').groupBy(['subreddit', 'TAVG']).agg(count('sentiment').alias('num_of_neu'))
    pd_hot_neu = grouped_hot_neg.toPandas()
    pd_hot_neu = pd_hot_neu.astype({'TAVG':'float', 'num_of_neu':'int'})
    x = np.array(pd_hot_neu['TAVG'])
    y = pd_hot_neu['num_of_neu']
    X = np.stack([x], axis=1)
    model.fit(X, y)
    pd_hot_neu['prediction'] = model.predict(X)
    #linear regression: hot and positive
    reg_hot_neu = stats.linregress(pd_hot_neu['TAVG'], y)
    pd_hot_neu['lin_predict'] = reg_hot_neu.slope * pd_hot_neu['TAVG'] + reg_hot_neu.intercept
    
    #Plot graph
    plt.figure(1, figsize=(15,5))
    plt.title("Hot weather and sentiment analysis")   
    
    #Plot 1
    plt.subplot(1, 3, 1)
    plt.plot(pd_hot_pos['TAVG'], pd_hot_pos['num_of_pos'], 'b.')
    plt.plot(pd_hot_pos['TAVG'], pd_hot_pos['prediction'], 'r-')
    plt.plot(pd_hot_pos['TAVG'], pd_hot_pos['lin_predict'], 'g-')
    plt.xlabel('Temperature')
    plt.ylabel('number of post')
    plt.title('linear regression of hot and positive')
    
    #Plot 2
    plt.subplot(1, 3, 2)
    plt.plot(pd_hot_neg['TAVG'], pd_hot_neg['num_of_neg'], 'b.')
    plt.plot(pd_hot_neg['TAVG'], pd_hot_neg['prediction'], 'r-')
    plt.plot(pd_hot_neg['TAVG'], pd_hot_neg['lin_predict'], 'g-')
    plt.xlabel('Temperature')
    plt.ylabel('number of post')
    plt.title('linear regression of hot and negative')
    
    #Plot 3
    plt.subplot(1, 3, 3)
    plt.plot(pd_hot_neu['TAVG'], pd_hot_neu['num_of_neu'], 'b.')
    plt.plot(pd_hot_neu['TAVG'], pd_hot_neu['prediction'], 'r-')
    plt.plot(pd_hot_neu['TAVG'], pd_hot_neu['lin_predict'], 'g-')
    plt.xlabel('Temperature')
    plt.ylabel('number of post')
    plt.title('linear regression of hot and neutral')
    
    plt.savefig('hot.png')
        
    #ANOVA test on hot weather
    #group by each day
    pd_hot_pos_grouped = hot_weather.filter(hot_weather['sentiment'] == 'positive').groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    pd_hot_neg_grouped = hot_weather.filter(hot_weather['sentiment'] == 'negative').groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    pd_hot_neu_grouped = hot_weather.filter(hot_weather['sentiment'] == 'neutral').groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    anova = stats.f_oneway(pd_hot_pos_grouped['count'], pd_hot_neg_grouped['count'], pd_hot_neu_grouped['count'])
    #post-hoc test: Tukey's HSD
    tukey_data = pd.concat([
        pd.DataFrame({'sentiment':'positive', 'count':pd_hot_pos_grouped['count'].values}),
        pd.DataFrame({'sentiment':'negative', 'count':pd_hot_neg_grouped['count'].values}),
        pd.DataFrame({'sentiment':'neutral', 'count':pd_hot_neu_grouped['count'].values}),
    ])
    posthoc = pairwise_tukeyhsd(tukey_data['count'], tukey_data['sentiment'], alpha=0.05)
        
    #result printing
    print('hot pos p-value: ', reg_hot_pos.pvalue)
    print('hot pos r-value: ', reg_hot_pos.rvalue)
    print('hot neg p-value: ', reg_hot_neg.pvalue)
    print('hot neg r-value: ', reg_hot_neg.rvalue)
    print('hot neu p-value: ', reg_hot_neu.pvalue)
    print('hot neu r-value: ', reg_hot_neu.rvalue)
    print('Anova p-value', anova.pvalue)
    print("Hot posthoc result:")
    print(posthoc)
    print()
        
main()