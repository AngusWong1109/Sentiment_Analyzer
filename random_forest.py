import sys
import matplotlib.pyplot as plt
from scipy import stats
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import col, count, to_date, dayofmonth, month, year, when
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('random forest').getOrCreate()
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
    data = joined_data.select('city', 'PRCP', 'SNOW', 'TMAX', 'TMIN', 'TAVG', 'sentiment')
    data = data.dropna()
    sentiment_indexer = StringIndexer(inputCol='sentiment', outputCol='sentiment_index')
    city_indexer = StringIndexer(inputCol='city', outputCol='city_index')
    indexed_data = sentiment_indexer.fit(data).transform(data)
    indexed_data = city_indexer.fit(indexed_data).transform(indexed_data)
    assembler = VectorAssembler(inputCols=['city_index', 'PRCP', 'SNOW', 'TMAX', 'TMIN', 'TAVG'], outputCol='features')
    processedData = assembler.transform(indexed_data)
    train_data, test_data = processedData.randomSplit([0.75, 0.25])
    randomForest = RandomForestClassifier(labelCol='sentiment_index', featuresCol='features', numTrees=20, maxDepth=10)
    rf_model = randomForest.fit(train_data)
    predictions = rf_model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol = 'sentiment_index', predictionCol='prediction', metricName='accuracy')
    print('Accuracy: ', evaluator.evaluate(predictions))
    
main()