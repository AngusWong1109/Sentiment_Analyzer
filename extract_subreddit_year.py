import sys
from pyspark.sql import SparkSession, functions, types
from datetime import datetime, timezone

spark = SparkSession.builder.appName('Split year').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

comments_schema = types.StructType([
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('body', types.StringType()),
    types.StructField('sentiment', types.StringType())
])

file_path = 'subreddit-output/subreddit-*'
output = 'subreddit-output_repartitioned/'

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

def main():
    
    reddit_data = spark.read.json(file_path, schema=comments_schema)

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

    for year in years:
        seattle_data_filtered = seattle_data.filter(seattle_data.year == year)
        losangeles_data_filtered = losangeles_data.filter(losangeles_data.year == year)
        atlanta_data_filtered = atlanta_data.filter(atlanta_data.year == year)
        calgary_data_filtered = calgary_data.filter(calgary_data.year == year)
        toronto_data_filtered = toronto_data.filter(toronto_data.year == year)
        montreal_data_filtered = montreal_data.filter(montreal_data.year == year)
        nyc_data_filtered = nyc_data.filter(nyc_data.year == year)
        boston_data_filtered = boston_data.filter(boston_data.year == year)
        chicago_data_filtered = chicago_data.filter(chicago_data.year == year)
        vancouver_data_filtered = vancouver_data.filter(vancouver_data.year == year)
        sanfran_data_filtered = sanfran_data.filter(sanfran_data.year == year)
        
        seattle_data_filtered.write.json(output+"seattle-"+str(year), compression='gzip', mode='overwrite')
        losangeles_data_filtered.write.json(output+"la-"+str(year), compression='gzip', mode='overwrite')
        atlanta_data_filtered.write.json(output+"atlanta-"+str(year), compression='gzip', mode='overwrite')
        calgary_data_filtered.write.json(output+"calgary-"+str(year), compression='gzip', mode='overwrite')
        toronto_data_filtered.write.json(output+"toronto-"+str(year), compression='gzip', mode='overwrite')
        montreal_data_filtered.write.json(output+"montreal-"+str(year), compression='gzip', mode='overwrite')
        nyc_data_filtered.write.json(output+"nyc-"+str(year), compression='gzip', mode='overwrite')
        boston_data_filtered.write.json(output+"boston-"+str(year), compression='gzip', mode='overwrite')
        chicago_data_filtered.write.json(output+"chicago-"+str(year), compression='gzip', mode='overwrite')
        vancouver_data_filtered.write.json(output+"vancouver-"+str(year), compression='gzip', mode='overwrite')
        sanfran_data_filtered.write.json(output+"sanfran-"+str(year), compression='gzip', mode='overwrite') 

if __name__=='__main__':
    # in_directory = sys.argv[1]
    # out_directory = sys.argv[2]
    main()