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

spark = SparkSession.builder.appName('cold_analysis').getOrCreate()
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
    
    joined_data_selected = joined_data.select(
        'station_id', 'date', weather.year, weather.month, weather.day, 'city', 'TMAX', 'TMIN', 'TAVG', 'T_label', 'subreddit', 'body', 'sentiment'
    )
    
    joined_data_selected = joined_data_selected.sort(['year', 'month', 'day'], ascending=[True, True, True])
    joined_data_selected = joined_data_selected.dropna()
    joined_data_selected = joined_data_selected.cache()
    
    """ 
    nyc = joined_data_selected.filter((joined_data_selected['city'] == "nyc"))
    la = joined_data_selected.filter((joined_data_selected['city'] == "LosAngeles"))
    boston = joined_data_selected.filter((joined_data_selected['city'] == "boston"))
    chicago = joined_data_selected.filter((joined_data_selected['city'] == "chicago"))
    seattle = joined_data_selected.filter((joined_data_selected['city'] == "Seattle"))
    atlanta = joined_data_selected.filter((joined_data_selected['city'] == "Atlanta"))
    sf = joined_data_selected.filter((joined_data_selected['city'] == "sanfrancisco"))
    toronto = joined_data_selected.filter((joined_data_selected['city'] == "toronto"))
    vancouver = joined_data_selected.filter((joined_data_selected['city'] == "vancouver"))
    calgary = joined_data_selected.filter((joined_data_selected['city'] == "Calgary"))
    montreal = joined_data_selected.filter((joined_data_selected['city'] == "montreal"))

    for i in argv:
        if i == "vancouver":
            cities_to_work_on.append(vancouver)
        elif i == "nyc":
            cities_to_work_on.append(nyc)
        elif i == "la":
            cities_to_work_on.append(la)
        elif i == "boston":
            cities_to_work_on.append(boston)
        elif i == "chicago":
            cities_to_work_on.append(chicago)
        elif i == "seattle":
            cities_to_work_on.append(seattle)
        elif i == "atlanta":
            cities_to_work_on.append(atlanta)
        elif i == "sf":
            cities_to_work_on.append(sf)
        elif i == "toronto":
            cities_to_work_on.append(toronto)
        elif i == "calgary":
            cities_to_work_on.append(calgary)
        elif i == "montreal":
            cities_to_work_on.append(montreal)
    """
    
    cold_weather = joined_data_selected.filter(
        joined_data_selected['T_label'] == 'cold'
    )
    hot_weather = joined_data_selected.filter(
        joined_data_selected['T_label'] == 'hot'
    )
    positive_post = joined_data_selected.filter(
        joined_data_selected['sentiment'] == 'positive'
    )
    negative_post = joined_data_selected.filter(
        joined_data_selected['sentiment'] == 'negative'
    )
    neutral_post = joined_data_selected.filter(
        joined_data_selected['sentiment'] == 'neutral'
    )
    
    model = make_pipeline(
        PolynomialFeatures(degree=4, include_bias=True),
        LinearRegression(fit_intercept=False)
    )
    
    #polynomial regression: cold and positive
    grouped_cold_pos = cold_weather.filter(cold_weather['sentiment'] == 'positive').groupBy(['subreddit', 'TAVG']).agg(count('sentiment').alias('num_of_pos'))
    pd_cold_pos = grouped_cold_pos.toPandas()
    pd_cold_pos = pd_cold_pos.astype({'TAVG':'float', 'num_of_pos':'int'})
    x = np.array(pd_cold_pos['TAVG'])
    y = pd_cold_pos['num_of_pos']
    X = np.stack([x], axis=1)
    model.fit(X, y)
    pd_cold_pos['prediction'] = model.predict(X)
    #linear regression: cold and positive
    reg_cold_pos = stats.linregress(pd_cold_pos['TAVG'], y)
    pd_cold_pos['lin_predict'] = reg_cold_pos.slope * pd_cold_pos['TAVG'] + reg_cold_pos.intercept    
    
    #polynomial regression: cold and negative
    grouped_cold_neg = cold_weather.filter(cold_weather['sentiment'] == 'negative').groupBy(['subreddit', 'TAVG']).agg(count('sentiment').alias('num_of_neg'))
    pd_cold_neg = grouped_cold_neg.toPandas()
    pd_cold_neg = pd_cold_neg.astype({'TAVG':'float', 'num_of_neg':'int'})
    x = np.array(pd_cold_neg['TAVG'])
    y = pd_cold_neg['num_of_neg']
    X = np.stack([x], axis=1)
    model.fit(X, y)
    pd_cold_neg['prediction'] = model.predict(X)
    #linear regression: cold and negative
    reg_cold_neg = stats.linregress(pd_cold_neg['TAVG'], y)
    pd_cold_neg['lin_predict'] = reg_cold_neg.slope * pd_cold_neg['TAVG'] + reg_cold_neg.intercept
    
    #polynomial regression: cold and neural
    grouped_cold_neg = cold_weather.filter(cold_weather['sentiment'] == 'neutral').groupBy(['subreddit', 'TAVG']).agg(count('sentiment').alias('num_of_neu'))
    pd_cold_neu = grouped_cold_neg.toPandas()
    pd_cold_neu = pd_cold_neu.astype({'TAVG':'float', 'num_of_neu':'int'})
    x = np.array(pd_cold_neu['TAVG'])
    y = pd_cold_neu['num_of_neu']
    X = np.stack([x], axis=1)
    model.fit(X, y)
    pd_cold_neu['prediction'] = model.predict(X)
    #linear regression: cold and neutral
    reg_cold_neu = stats.linregress(pd_cold_neu['TAVG'], y)
    pd_cold_neu['lin_predict'] = reg_cold_neu.slope * pd_cold_neu['TAVG'] + reg_cold_neu.intercept
    
    #Plot graph
    plt.figure(1, figsize=(15,5))
    plt.title("Cold weather and sentiment analysis") 
    
    #Plot 1
    plt.subplot(1, 3, 1)
    plt.plot(pd_cold_pos['TAVG'], pd_cold_pos['num_of_pos'], 'b.')
    plt.plot(pd_cold_pos['TAVG'], pd_cold_pos['prediction'], 'r-')
    plt.plot(pd_cold_pos['TAVG'], pd_cold_pos['lin_predict'], 'g-')
    plt.xlabel('Temperature')
    plt.ylabel('number of post')
    plt.title('linear regression of cold and positive')
    
    #Plot 2
    plt.subplot(1, 3, 2)
    plt.plot(pd_cold_neg['TAVG'], pd_cold_neg['num_of_neg'], 'b.')
    plt.plot(pd_cold_neg['TAVG'], pd_cold_neg['prediction'], 'r-')
    plt.plot(pd_cold_neg['TAVG'], pd_cold_neg['lin_predict'], 'g-')
    plt.xlabel('Temperature')
    plt.ylabel('number of post')
    plt.title('linear regression of cold and negative')
    
    #Plot 3
    plt.subplot(1, 3, 3)
    plt.plot(pd_cold_neu['TAVG'], pd_cold_neu['num_of_neu'], 'b.')
    plt.plot(pd_cold_neu['TAVG'], pd_cold_neu['prediction'], 'r-')
    plt.plot(pd_cold_neu['TAVG'], pd_cold_neu['lin_predict'], 'g-')
    plt.xlabel('Temperature')
    plt.ylabel('number of post')
    plt.title('linear regression of cold and neutral')
        
    plt.savefig('cold.png')
       
    #chi-square test
    pd_cold_pos = cold_weather.filter(cold_weather['sentiment'] == 'positive').agg(count('sentiment').alias('count')).toPandas()
    pd_cold_neg = cold_weather.filter(cold_weather['sentiment'] == 'negative').agg(count('sentiment').alias('count')).toPandas()
    pd_cold_neu = cold_weather.filter(cold_weather['sentiment'] == 'neutral').agg(count('sentiment').alias('count')).toPandas()
    pd_hot_pos = hot_weather.filter(hot_weather['sentiment'] == 'positive').agg(count('sentiment').alias('count')).toPandas()
    pd_hot_neg = hot_weather.filter(hot_weather['sentiment'] == 'negative').agg(count('sentiment').alias('count')).toPandas()
    pd_hot_neu = hot_weather.filter(hot_weather['sentiment'] == 'neutral').agg(count('sentiment').alias('count')).toPandas()
    pos_cold_count = pd_cold_pos['count'][0]
    neg_cold_count = pd_cold_neg['count'][0]
    neu_cold_count = pd_cold_neu['count'][0]
    pos_hot_count = pd_hot_pos['count'][0]
    neg_hot_count = pd_hot_neg['count'][0]
    neu_hot_count = pd_hot_neu['count'][0]
    contingency = [[pos_cold_count, pos_hot_count], [neg_cold_count, neg_hot_count], [neu_cold_count, neu_hot_count]]
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
    #ANOVA test on cold weather
    #group by each day
    pd_cold_pos_grouped = cold_weather.filter(cold_weather['sentiment'] == 'positive').groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    pd_cold_neg_grouped = cold_weather.filter(cold_weather['sentiment'] == 'negative').groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    pd_cold_neu_grouped = cold_weather.filter(cold_weather['sentiment'] == 'neutral').groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    anova = stats.f_oneway(pd_cold_pos_grouped['count'], pd_cold_neg_grouped['count'], pd_cold_neu_grouped['count'])
    print('Anova p-value', anova.pvalue)
    #post-hoc test: Tukey's HSD
    tukey_data = pd.concat([
        pd.DataFrame({'sentiment':'positive', 'count':pd_cold_pos_grouped['count'].values}),
        pd.DataFrame({'sentiment':'negative', 'count':pd_cold_neg_grouped['count'].values}),
        pd.DataFrame({'sentiment':'neutral', 'count':pd_cold_neu_grouped['count'].values}),
    ])
    posthoc = pairwise_tukeyhsd(tukey_data['count'], tukey_data['sentiment'], alpha=0.05)
        
    #U-test
    pd_pos_grouped = positive_post.groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    pd_neg_grouped = negative_post.groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
    pd_neu_grouped = neutral_post.groupBy(['year', 'month', 'day']).agg(count('sentiment').alias('count')).toPandas()
        
    #result printing
    print('cold pos p-value: ', reg_cold_pos.pvalue)
    print('cold pos r-value: ', reg_cold_pos.rvalue)
    print('cold neg p-value: ', reg_cold_neg.pvalue)
    print('cold neg r-value: ', reg_cold_neg.rvalue)
    print('cold neu p-value: ', reg_cold_neu.pvalue)
    print('cold neu r-value: ', reg_cold_neu.rvalue)
    print('chi-square test pvalue = ', p)
    print("Cold posthoc result:")
    print(posthoc)
    print('U test:')
    print('Positive vs negative U-test result:', stats.mannwhitneyu(pd_pos_grouped['count'], pd_neg_grouped['count']))
    print('Neutral vs negative U-test result:', stats.mannwhitneyu(pd_neu_grouped['count'], pd_neg_grouped['count']))
    print('Positive vs neutral U-test result:', stats.mannwhitneyu(pd_pos_grouped['count'], pd_neu_grouped['count']))
    print()
        
main()