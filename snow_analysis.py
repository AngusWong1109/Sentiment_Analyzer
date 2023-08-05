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

def main():
    reddit_path = 'subreddit-output/subreddit-*'
    weather_path = 'weather-output/weather-*'
    weather_sample_filepath = 'sample_dataset/weather_output/weather-*'
    reddit_sample_filepath = 'sample_dataset/reddit_output/part-*'
    # print(reddit_path)

    if sys.argv[1] == 'whole':
        reddit_data = spark.read.json(reddit_path, schema=comments_schema)
        weather_data = spark.read.csv(weather_path, schema=weather_schema)
    else:
        reddit_data = spark.read.json(reddit_sample_filepath, schema=comments_schema)
        weather_data = spark.read.csv(weather_sample_filepath, schema=weather_schema)

    analyze_sentiment = functions.udf(get_sentiment, returnType=types.StringType())
    reddit_data = reddit_data.withColumn('sentiment', analyze_sentiment(reddit_data['body']))
    reddit_data.cache()

    weather_data.show()

    snow_data = weather_data.filter((weather_data['SNOW'].isNotNull()))
    snow_data = snow_data.drop(snow_data['date'], snow_data['PRCP'], snow_data['SNWD'], snow_data['TMAX'], snow_data['TMIN'], snow_data['TAVG'], snow_data['T_label'])
    snow_data.cache()
    snow_data.show()

    reddit_count = reddit_data.groupBy(['year','month','day','sentiment']).count()
    combined = reddit_count.join(snow_data, ['year','month','day']).orderBy(['year','month','day','sentiment'])
    combined = combined.withColumn('has_snow',when(combined['SNOW'] > 0, True).otherwise(False))
    combined.cache()

    ### PANDAS DATA FRAME ###
    combined_pd = combined.toPandas()
    print(combined_pd)

    neg_combined_pd = combined_pd.loc[combined_pd['sentiment']=='negative']
    pos_combined_pd = combined_pd.loc[combined_pd['sentiment']=='positive']
    neutral_combined_pd = combined_pd.loc[combined_pd['sentiment']=='neutral']

    neg_snow_pd = combined_pd.loc[(combined_pd['sentiment']=='negative') & (combined_pd['has_snow']==True)]
    pos_snow_pd = combined_pd.loc[(combined_pd['sentiment']=='positive') & (combined_pd['has_snow']==True)]
    neutral_snow_pd = combined_pd.loc[(combined_pd['sentiment']=='neutral') & (combined_pd['has_snow']==True)]

    neg_no_snow_pd = combined_pd.loc[(combined_pd['sentiment']=='negative') & (combined_pd['has_snow']==False)]
    pos_no_snow_pd = combined_pd.loc[(combined_pd['sentiment']=='positive') & (combined_pd['has_snow']==False)]
    neutral_no_snow_pd = combined_pd.loc[(combined_pd['sentiment']=='neutral') & (combined_pd['has_snow']==False)]

    
    # #### LINEAR REGRESSION PLOT & CALCULATIONS ####
    
    pos_regression = stats.linregress(pos_combined_pd['count'],pos_combined_pd['SNOW'])
    pos_combined_pd['prediction'] = (pos_regression.slope * pos_combined_pd['count']) + pos_regression.intercept

    neg_regression = stats.linregress(neg_combined_pd['count'],neg_combined_pd['SNOW'])
    neg_combined_pd['prediction'] = (neg_regression.slope * neg_combined_pd['count']) + neg_regression.intercept

    neut_regression = stats.linregress(neutral_combined_pd['count'],neutral_combined_pd['SNOW'])
    neutral_combined_pd['prediction'] = (neut_regression.slope * neutral_combined_pd['count']) + neut_regression.intercept

    print("LINEAR REGRESSION: Positive lin reg p-value:", pos_regression.pvalue, "r-value:",pos_regression.rvalue)
    print("LINEAR REGRESSION: Negative lin reg p-value:", neg_regression.pvalue, "r-value:",neg_regression.rvalue)
    print("LINEAR REGRESSION: Neutral lin reg p-value:", neut_regression.pvalue, "r-value:",neut_regression.rvalue)


    ### FIGURE 1 - SCATTER PLOT
    plt.figure(1, figsize=(15,5))
    # plt.figure(1)
    plt.title("Snow and Sentiment Correlation")

    # Plot 1
    plt.subplot(1,3,1)
    plt.plot(pos_combined_pd['count'],pos_combined_pd['SNOW'], 'b.', alpha=0.5)
    plt.plot(pos_combined_pd['count'],pos_combined_pd['prediction'], 'r', linewidth=2)
    plt.xlabel("# of reddit comments (daily)")
    plt.ylabel("Amount of daily snow (in mm)")
    plt.title("Positive Sentiment & Snow Correlation")
    # plt.semilogx()
    # plt.semilogy()

    # Plot 2
    plt.subplot(1,3,2)
    plt.plot(neg_combined_pd['count'],neg_combined_pd['SNOW'], 'c.', alpha=0.5)
    plt.plot(neg_combined_pd['count'],neg_combined_pd['prediction'], 'r', linewidth=2)
    plt.xlabel("# of reddit comments (daily)")
    plt.ylabel("Amount of daily snow (in mm)")
    plt.title("Negative Sentiment & Snow Correlation")
    # plt.semilogx()
    # plt.semilogy()

    # Plot 3
    plt.subplot(1,3,3)
    plt.plot(neutral_combined_pd['count'],neutral_combined_pd['SNOW'], 'g.', alpha=0.5)
    plt.plot(neutral_combined_pd['count'],neutral_combined_pd['prediction'], 'r', linewidth=2)
    plt.xlabel("# of reddit comments (daily)")
    plt.ylabel("Amount of daily snow (in mm)")
    plt.title("Neutral Sentiment & Snow Correlation")
    # plt.semilogx()
    # plt.semilogy()

    # plt.legend(['positive','negative','neutral'])
    plt.savefig('allcities-snow-scatter.png')
    # plt.show()

    ### FIGURE 2 - HISTOGRAM
    plt.figure(2)
    plt.hist([pos_combined_pd['count'], neg_combined_pd['count'], neutral_combined_pd['count']])
    plt.legend(['positive','negative', 'neutral'])
    plt.xlabel("Frequency of # of daily reddit comments")
    plt.ylabel("# of positive/negative/neutral reddit comments per day")
    # plt.show()
    plt.savefig('allcities-snow-hist.png')
    

    #### CHI SQUARE ####
    combined_pd_posneg = combined_pd.where(combined_pd['sentiment']!='neutral').dropna()

    contingency = pd.pivot_table(combined_pd_posneg, values='count', index=['has_snow'], columns=['sentiment'], aggfunc=np.sum)
    print("Contingency Table:")
    print(contingency)

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("Chi P-value:",p)

    ### Chi square including neutral comments
    combined_pd_all = combined_pd.dropna()

    contingency_all = pd.pivot_table(combined_pd_all, values='count', index=['has_snow'], columns=['sentiment'], aggfunc=np.sum)
    print("Contingency All Table:")
    print(contingency_all )

    chi2_all, p_all, dof_all, expected_all = stats.chi2_contingency(contingency_all)
    print("Chi All P-value:",p_all)

    #### ANOVA AND TUKEYHSD ANALYSIS ####
    anova_snow = stats.f_oneway(neg_snow_pd['count'], pos_snow_pd['count'], neutral_snow_pd['count'])
    print("ANOVA - Snow Data P-value:", anova_snow.pvalue)

    tukey_data_snow = pd.concat([
        pd.DataFrame({'sentiment':'negative', 'count':neg_snow_pd['count'].values}),
        pd.DataFrame({'sentiment':'positive', 'count':pos_snow_pd['count'].values}),
        pd.DataFrame({'sentiment':'neutral', 'count':neutral_snow_pd['count'].values}),
    ])

    posthoc_snow = pairwise_tukeyhsd(tukey_data_snow['count'], tukey_data_snow['sentiment'], alpha=0.05)
    print("Tukey HSD - Snow Data:")
    print(posthoc_snow)

    anova_combined = stats.f_oneway(neg_snow_pd['count'], pos_snow_pd['count'], neutral_snow_pd['count'], neg_no_snow_pd['count'], pos_no_snow_pd['count'], neutral_no_snow_pd['count'])
    print("ANOVA - Combined Data P-value:", anova_combined.pvalue)

    tukey_data_combined = pd.concat([
        pd.DataFrame({'sentiment':'snow-negative', 'count':neg_snow_pd['count'].values}),
        pd.DataFrame({'sentiment':'snow-positive', 'count':pos_snow_pd['count'].values}),
        pd.DataFrame({'sentiment':'snow-neutral', 'count':neutral_snow_pd['count'].values}),
        pd.DataFrame({'sentiment':'noSnow-negative', 'count':neg_no_snow_pd['count'].values}),
        pd.DataFrame({'sentiment':'noSnow-positive', 'count':pos_no_snow_pd['count'].values}),
        pd.DataFrame({'sentiment':'noSnow-neutral', 'count':neutral_no_snow_pd['count'].values}),
    ])

    posthoc_combined = pairwise_tukeyhsd(tukey_data_combined['count'], tukey_data_combined['sentiment'], alpha=0.05)
    print("Tukey HSD - Combined Data:")
    print(posthoc_combined)



if __name__=='__main__':
    # city = sys.argv[1]
    # out_directory = sys.argv[2]
    main()