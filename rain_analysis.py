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
    reddit_data = reddit_data.withColumn('sentiment', analyze_sentiment(reddit_data['body']))
    reddit_data.cache()

    rainy_days = weather_data.withColumn('PRCP', weather_data['PRCP']/10).filter((weather_data['PRCP'].isNotNull()))
    rainy_days = rainy_days.drop(rainy_days['date'], rainy_days['SNOW'], rainy_days['SNWD'], rainy_days['TMAX'], rainy_days['TMIN'], rainy_days['TAVG'], rainy_days['T_label'])
    rainy_days.cache()


    reddit_count = reddit_data.groupBy(['year','month','day','sentiment']).count()
    combined = reddit_count.join(rainy_days, ['year','month','day']).orderBy(['year','month','day','sentiment'])
    combined = combined.withColumn('has_rain',when(combined.PRCP > 0, True).otherwise(False))
    combined.cache()

    ### PANDAS DATA FRAME ###
    combined_pd = combined.toPandas()
    print(combined_pd)

    neg_combined_pd = combined_pd.loc[combined_pd['sentiment']=='negative']
    pos_combined_pd = combined_pd.loc[combined_pd['sentiment']=='positive']
    neutral_combined_pd = combined_pd.loc[combined_pd['sentiment']=='neutral']

    neg_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='negative') & (combined_pd['has_rain']==True)]
    pos_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='positive') & (combined_pd['has_rain']==True)]
    neutral_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='neutral') & (combined_pd['has_rain']==True)]

    neg_no_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='negative') & (combined_pd['has_rain']==False)]
    pos_no_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='positive') & (combined_pd['has_rain']==False)]
    neutral_no_rain_pd = combined_pd.loc[(combined_pd['sentiment']=='neutral') & (combined_pd['has_rain']==False)]

    
    #### LINEAR REGRESSION PLOT & CALCULATIONS ####
    

    pos_regression = stats.linregress(pos_combined_pd['count'],pos_combined_pd['PRCP'])
    pos_combined_pd['prediction'] = (pos_regression.slope * pos_combined_pd['count']) + pos_regression.intercept

    neg_regression = stats.linregress(neg_combined_pd['count'],neg_combined_pd['PRCP'])
    neg_combined_pd['prediction'] = (neg_regression.slope * neg_combined_pd['count']) + neg_regression.intercept


    neut_regression = stats.linregress(neutral_combined_pd['count'],neutral_combined_pd['PRCP'])
    neutral_combined_pd['prediction'] = (neut_regression.slope * neutral_combined_pd['count']) + neut_regression.intercept

    print("LINEAR REGRESSION: Positive lin reg p-value:", pos_regression.pvalue, "r-value:",pos_regression.rvalue)
    print("LINEAR REGRESSION: Negative lin reg p-value:", neg_regression.pvalue, "r-value:",neg_regression.rvalue)
    print("LINEAR REGRESSION: Neutral lin reg p-value:", neut_regression.pvalue, "r-value:",neut_regression.rvalue)


    ### FIGURE 1 - SCATTER PLOT
    plt.figure(1, figsize=(15,5))
    # plt.figure(1)
    plt.title("Rain and Sentiment Correlation")

    # Plot 1
    plt.subplot(1,3,1)
    plt.plot(pos_combined_pd['count'],pos_combined_pd['PRCP'], 'b.')
    plt.plot(pos_combined_pd['count'],pos_combined_pd['prediction'], 'r', linewidth=2)
    plt.xlabel("# of reddit comments (daily)")
    plt.ylabel("Amount of daily rain (in mm)")
    plt.title("Positive Sentiment & Rain Correlation")

    # Plot 2
    plt.subplot(1,3,2)
    plt.plot(neg_combined_pd['count'],neg_combined_pd['PRCP'], 'c.')
    plt.plot(neg_combined_pd['count'],neg_combined_pd['prediction'], 'r', linewidth=2)
    plt.xlabel("# of reddit comments (daily)")
    plt.ylabel("Amount of daily rain (in mm)")
    plt.title("Negative Sentiment & Rain Correlation")

    # Plot 3
    plt.subplot(1,3,3)
    plt.plot(neutral_combined_pd['count'],neutral_combined_pd['PRCP'], 'g.')
    plt.plot(neutral_combined_pd['count'],neutral_combined_pd['prediction'], 'r', linewidth=2)
    plt.xlabel("# of reddit comments (daily)")
    plt.ylabel("Amount of daily rain (in mm)")
    plt.title("Neutral Sentiment & Rain Correlation")

    # plt.legend(['positive','negative','neutral'])
    plt.savefig(city+'-scatter.png')
    # plt.show()

    ### FIGURE 2 - HISTOGRAM
    plt.figure(2)
    plt.hist([pos_combined_pd['count'], neg_combined_pd['count'], neutral_combined_pd['count']])
    plt.legend(['positive','negative', 'neutral'])
    plt.xlabel("Frequency of # of daily reddit comments")
    plt.ylabel("# of positive/negative/neutral reddit comments per day")
    # plt.show()
    plt.savefig(city+'-hist.png')

    #### CHI SQUARE ####
    combined_pd_posneg = combined_pd.where(combined_pd['sentiment']!='neutral').dropna()

    contingency = pd.pivot_table(combined_pd_posneg, values='count', index=['has_rain'], columns=['sentiment'], aggfunc=np.sum)
    print("Contingency Table:")
    print(contingency)

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("Chi P-value:",p)

    ### Chi square including neutral comments
    combined_pd_all = combined_pd.dropna()

    contingency_all = pd.pivot_table(combined_pd_all, values='count', index=['has_rain'], columns=['sentiment'], aggfunc=np.sum)
    print("Contingency All Table:")
    print(contingency_all )

    chi2_all, p_all, dof_all, expected_all = stats.chi2_contingency(contingency_all)
    print("Chi All P-value:",p_all)

    ### Sample amount for chi-square
    combined_pd_posneg_sample = combined_pd_posneg.sample(frac=0.5, random_state=1)

    contingency_sample = pd.pivot_table(combined_pd_posneg_sample, values='count', index=['has_rain'], columns=['sentiment'], aggfunc=np.sum)
    print("Contingency Sample Table:")
    print(contingency_sample)

    chi2_sample, p_sample, dof_sample, expected_sample = stats.chi2_contingency(contingency_sample)
    print("Chi Sample P-value:",p_sample)


    #### ANOVA AND TUKEYHSD ANALYSIS ####
    anova_rain = stats.f_oneway(neg_rain_pd['count'], pos_rain_pd['count'], neutral_rain_pd['count'])
    print("ANOVA - Rain Data P-value:", anova_rain.pvalue)

    tukey_data_rain = pd.concat([
        pd.DataFrame({'sentiment':'negative', 'count':neg_rain_pd['count'].values}),
        pd.DataFrame({'sentiment':'positive', 'count':pos_rain_pd['count'].values}),
        pd.DataFrame({'sentiment':'neutral', 'count':neutral_rain_pd['count'].values}),
    ])

    posthoc_rain = pairwise_tukeyhsd(tukey_data_rain['count'], tukey_data_rain['sentiment'], alpha=0.05)
    print("Tukey HSD - Rain Data:")
    print(posthoc_rain)

    anova_combined = stats.f_oneway(neg_rain_pd['count'], pos_rain_pd['count'], neutral_rain_pd['count'], neg_no_rain_pd['count'], pos_no_rain_pd['count'], neutral_no_rain_pd['count'])
    print("ANOVA - Combined Data P-value:", anova_combined.pvalue)

    tukey_data_combined = pd.concat([
        pd.DataFrame({'sentiment':'rain-negative', 'count':neg_rain_pd['count'].values}),
        pd.DataFrame({'sentiment':'rain-positive', 'count':pos_rain_pd['count'].values}),
        pd.DataFrame({'sentiment':'rain-neutral', 'count':neutral_rain_pd['count'].values}),
        pd.DataFrame({'sentiment':'noRain-negative', 'count':neg_no_rain_pd['count'].values}),
        pd.DataFrame({'sentiment':'noRain-positive', 'count':pos_no_rain_pd['count'].values}),
        pd.DataFrame({'sentiment':'noRain-neutral', 'count':neutral_no_rain_pd['count'].values}),
    ])

    posthoc_combined = pairwise_tukeyhsd(tukey_data_combined['count'], tukey_data_combined['sentiment'], alpha=0.05)
    print("Tukey HSD - Combined Data:")
    print(posthoc_combined)


    

if __name__=='__main__':
    city = sys.argv[1]
    # out_directory = sys.argv[2]
    main(city)