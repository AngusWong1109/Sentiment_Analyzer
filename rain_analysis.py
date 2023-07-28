import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.functions import when
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
    reddit_data = reddit_data.withColumn('sentiment', analyze_sentiment(reddit_data['body']))

    rainy_days = weather_data.filter((weather_data['element'] == "PRCP"))
    reddit_count = reddit_data.groupBy(['year','month','day','sentiment']).count()
    combined = reddit_count.join(rainy_days, ['year','month','day']).drop(rainy_days['obs_time'],rainy_days['date']).orderBy(['year','month','day','sentiment'])
    combined = combined.withColumn('has_rain',when(combined.data_value > 0, True).otherwise(False))
    combined.cache()



    # FOR IF WE WILL USE SPARK DATAFRAME
    # combined_negative = combined.filter((combined['sentiment'] == "negative"))
    # combined_neutral = combined.filter((combined['sentiment'] == "neutral"))
    # combined_positive = combined.filter((combined['sentiment'] == "positive"))

    # Put data in Pandas dataframe
    combined_pd = combined.toPandas()
    combined_pd['has_rain'] = np.where(combined_pd['data_value']>0, True, False)

    #Rain data and no rain data
    combined_rain = combined.filter((combined['has_rain']==True))
    combined_no_rain = combined.filter((combined['has_rain']==False))
    # combined_pd_rain = combined_pd.loc[combined_pd['has_rain']==True]
    # combined_pd_no_rain = combined_pd.loc[combined_pd['has_rain']==False]


    # rain = combined_pd_rain[['sentiment','count']].groupby('sentiment').mean()
    # print(rain)

    # no_rain = combined_pd_no_rain[['sentiment','count']].groupby('sentiment').mean()
    # print(no_rain)

    # Rain and no rain data combined split into positive, negative, neutral
    neg_combined = combined.filter((combined['sentiment']=='negative'))
    pos_combined = combined.filter((combined['sentiment']=='positive'))
    neut_combined = combined.filter((combined['sentiment']=='neutral'))

    neg_combined_pd = neg_combined.toPandas()
    pos_combined_pd = pos_combined.toPandas()
    neut_combined_pd = neut_combined.toPandas()

    # negative_pd_combined = combined_pd.loc[combined_pd['sentiment']=='negative']
    # positive_pd_combined = combined_pd.loc[combined_pd['sentiment']=='positive']
    # neutral_pd_combined = combined_pd.loc[combined_pd['sentiment']=='neutral']

    # Rain data combined split into positive, negative, neutral
    neg_rain = combined_rain.filter((combined_rain['sentiment']=='negative'))
    pos_rain = combined_rain.filter((combined_rain['sentiment']=='positive'))
    neut_rain = combined_rain.filter((combined_rain['sentiment']=='neutral'))
    # negative_pd_rain = combined_pd_rain.loc[combined_pd_rain['sentiment']=='negative']
    # positive_pd_rain = combined_pd_rain.loc[combined_pd_rain['sentiment']=='positive']
    # neutral_pd_rain = combined_pd_rain.loc[combined_pd_rain['sentiment']=='neutral']

    # No rain data combined split into positive, negative, neutral
    neg_no_rain = combined_no_rain.filter((combined_no_rain['sentiment']=='negative'))
    pos_no_rain = combined_no_rain.filter((combined_no_rain['sentiment']=='positive'))
    neut_no_rain = combined_no_rain.filter((combined_no_rain['sentiment']=='neutral'))
    # negative_pd_no_rain = combined_pd_no_rain.loc[combined_pd_no_rain['sentiment']=='negative']
    # positive_pd_no_rain = combined_pd_no_rain.loc[combined_pd_no_rain['sentiment']=='positive']
    # neutral_pd_no_rain = combined_pd_no_rain.loc[combined_pd_no_rain['sentiment']=='neutral']

    #### LINEAR REGRESSION PLOT & CALCULATIONS ####

    # pos_combined_count = pos_combined.select('count')
    # pos_combined_rain_amt = pos_combined.select('data_value')

    # neg_combined_count = neg_combined.select('count')
    # neg_combined_rain_amt = neg_combined.select('data_value')

    plt.plot(pos_combined_pd['count'],pos_combined_pd['data_value'], 'b.')
    plt.plot(neg_combined_pd['count'],neg_combined_pd['data_value'], 'r.')
    plt.plot(neut_combined_pd['count'],neut_combined_pd['data_value'], 'g.')
    plt.legend(['positive','negative','neutral'])
    plt.savefig(city+'-scatter.png')
    # plt.plot(neutral_pd_combined['count'],neutral_pd_combined['data_value'], 'g.')
    # plt.show()

    pos_reg = stats.linregress(pos_combined_pd['count'],pos_combined_pd['data_value'])
    neg_reg = stats.linregress(neg_combined_pd['count'],neg_combined_pd['data_value'])

    print("Positive lin reg p-value:", pos_reg.pvalue, "r-value:",pos_reg.rvalue)
    print("Negative lin reg p-value:", neg_reg.pvalue, "r-value:",neg_reg.rvalue)

    # plt.hist([pos_combined_pd['count'], neg_combined_pd['count']])
    # plt.legend(['positive','negative'])
    # plt.show()
    # plt.savefig('vancouver-hist.png')

    #### CHI SQUARE ####
    # combined.show()

    # sentiment_count = combined.select(['sentiment', 'count', 'year']).groupBy(['sentiment','year']).sum().toPandas()
    # print(sentiment_count)

    # pd.DataFrame()

    # print(combined_pd)
    combined_pd_posneg = combined_pd.where(combined_pd['sentiment']!='neutral').dropna()
    print(combined_pd_posneg)

    

    contingency = pd.pivot_table(combined_pd_posneg, values='count', index=['has_rain'], columns=['sentiment'], aggfunc=np.sum)
    print("Contingency Table:")
    print(contingency)

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("Chi P-value:",p)


    combined_pd_posneg_sample = combined_pd_posneg.sample(frac=0.5, random_state=1)
    print(combined_pd_posneg_sample)

    contingency_sample = pd.pivot_table(combined_pd_posneg_sample, values='count', index=['has_rain'], columns=['sentiment'], aggfunc=np.sum)
    print("Contingency Sample Table:")
    print(contingency_sample)

    chi2_sample, p_sample, dof_sample, expected_sample = stats.chi2_contingency(contingency_sample)
    print("Chi Sample P-value:",p_sample)


    #### ANOVA AND TUKEYHSD ANALYSIS ####
    # anova_rain = stats.f_oneway(negative_pd_rain['count'], positive_pd_rain['count'], neutral_pd_rain['count'])
    # print(anova_rain.pvalue)

    # anova_no_rain = stats.f_oneway(negative_pd_no_rain['count'], positive_pd_no_rain['count'], neutral_pd_no_rain['count'])
    # print(anova_no_rain.pvalue)

    # tukey_data_rain = pd.concat([
    #     pd.DataFrame({'sentiment':'negative', 'count':negative_pd_rain['count'].values}),
    #     pd.DataFrame({'sentiment':'positive', 'count':positive_pd_rain['count'].values}),
    #     pd.DataFrame({'sentiment':'neutral', 'count':neutral_pd_rain['count'].values}),
    # ])


    # posthoc_rain = pairwise_tukeyhsd(tukey_data_rain['count'], tukey_data_rain['sentiment'], alpha=0.05)
    # print(posthoc_rain)

    # tukey_data_no_rain = pd.concat([
    #     pd.DataFrame({'sentiment':'negative', 'count':negative_pd_no_rain['count'].values}),
    #     pd.DataFrame({'sentiment':'positive', 'count':positive_pd_no_rain['count'].values}),
    #     pd.DataFrame({'sentiment':'neutral', 'count':neutral_pd_no_rain['count'].values}),
    # ])

    # posthoc_no_rain = pairwise_tukeyhsd(tukey_data_no_rain['count'], tukey_data_no_rain['sentiment'], alpha=0.05)
    # print(posthoc_no_rain)

    


if __name__=='__main__':
    city = sys.argv[1]
    # out_directory = sys.argv[2]
    main(city)