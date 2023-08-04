import sys
from pyspark.sql import SparkSession, functions, types
from datetime import datetime, timezone


spark = SparkSession.builder.appName('Analyze Reddit').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

comments_schema = types.StructType([
    types.StructField('subreddit', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('sentiment', types.StringType()),
])

file_path = 'reddit-subset/comments/part*'
output = 'subreddit-output/subreddit-'


def main():
    data = spark.read.json(file_path, schema=comments_schema)

    reddit_data = data.withColumn('timestamp', functions.from_unixtime(data['created_utc'],"yyyy-MM-dd"))
    reddit_data = reddit_data.withColumn('year', functions.year(reddit_data['timestamp']))\
                            .withColumn('month', functions.month(reddit_data['timestamp']))\
                            .withColumn('day', functions.dayofmonth(reddit_data['timestamp']))

    reddit_data = reddit_data.filter((reddit_data['body'] != "[deleted]") & (reddit_data['body'] != "[removed]"))\
        .select(reddit_data['subreddit'], reddit_data['year'], reddit_data['month'],
                reddit_data['day'], reddit_data['body'])

    reddit_data = reddit_data.cache()

    seattle_data = reddit_data.where(reddit_data['subreddit']=="Seattle")
    losangeles_data = reddit_data.where(reddit_data['subreddit'] == "LosAngeles")
    atlanta_data = reddit_data.where(reddit_data['subreddit'] == "Atlanta")
    calgary_data = reddit_data.where(reddit_data['subreddit'] == "Calgary")
    toronto_data = reddit_data.where(reddit_data['subreddit'] == "toronto")
    montreal_data = reddit_data.where(reddit_data['subreddit'] == "montreal")
    nyc_data = reddit_data.where(reddit_data['subreddit'] == "nyc")
    boston_data = reddit_data.where(reddit_data['subreddit'] == "boston")
    chicago_data = reddit_data.where(reddit_data['subreddit'] == "chicago")
    vancouver_data = reddit_data.where(reddit_data['subreddit'] == "vancouver")
    sanfran_data = reddit_data.where(reddit_data['subreddit'] == "sanfrancisco")

    seattle_data.write.json(output+"seattle", compression='gzip', mode='overwrite')
    losangeles_data.write.json(output + "la", compression='gzip', mode='overwrite')
    atlanta_data.write.json(output + "atlanta", compression='gzip', mode='overwrite')
    calgary_data.write.json(output + "calgary", compression='gzip', mode='overwrite')
    toronto_data.write.json(output + "toronto", compression='gzip', mode='overwrite')
    montreal_data.write.json(output + "montreal", compression='gzip', mode='overwrite')
    nyc_data.write.json(output + "nyc", compression='gzip', mode='overwrite')
    boston_data.write.json(output + "boston", compression='gzip', mode='overwrite')
    chicago_data.write.json(output + "chicago", compression='gzip', mode='overwrite')
    vancouver_data.write.json(output + "vancouver", compression='gzip', mode='overwrite')
    sanfran_data.write.json(output + "sf", compression='gzip', mode='overwrite')


    reddit_data.show()

if __name__=='__main__':
    # in_directory = sys.argv[1]
    # out_directory = sys.argv[2]
    main()