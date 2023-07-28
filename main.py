import sys
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import udf
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('GHCN data').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

output = 'weather-output/weather-'
weather_file_path = [
    "/courses/datasets/ghcn/2015.csv.gz",
    "/courses/datasets/ghcn/2016.csv.gz",
    "/courses/datasets/ghcn/2017.csv.gz",
    "/courses/datasets/ghcn/2018.csv.gz",
    "/courses/datasets/ghcn/2019.csv.gz",
    "/courses/datasets/ghcn/2020.csv.gz",
    "/courses/datasets/ghcn/2021.csv.gz",
]

weather_file_path_local = [
    "2019.csv.gz",
    "2020.csv.gz",
    "2021.csv.gz",
]

ids = {
    'USW00094728': 'New York',
    'USW00023174': 'Los Angeles',
    'USW00014739': 'Boston',
    'USW00014819': 'Chicago',
    'USW00024233': 'Seattle',
    'USW00053863': 'Atlanta',
    'USW00023234': 'San Francisco',
    'CA006158731': 'Toronto',
    'CA001108395': 'Vancouver',
    'CA003031092': 'Calgary',
    'CA007025251': 'Montreal'
}


stations = [
    'USW00094728',
    'USW00023174',
    'USW00014739',
    'USW00014819',
    'USW00024233',
    'USW00053863',
    'USW00023234',
    'CA006158731',
    'CA001108395',
    'CA003031092',
    'CA007025251',
]

weather_schema = types.StructType([
    types.StructField('station_id', types.StringType()),
    types.StructField('date', types.IntegerType()),
    types.StructField('element', types.StringType()),
    types.StructField('data_value', types.IntegerType()),
    types.StructField('m_flag', types.StringType()),
    types.StructField('q_flag', types.StringType()),
    types.StructField('s_flag', types.StringType()),
    types.StructField('obs_time', types.IntegerType()),
])

def main():
    weather_data = spark.read.csv(weather_file_path, schema=weather_schema)
    ids_df = spark.createDataFrame(list(ids.items()), ["station_id", "city"])
    weather_data = weather_data.filter(
        weather_data['station_id'].isin(stations)
    )
    weather_data = weather_data.withColumn('year', (col('date')/10000).cast('Integer')).withColumn('month', ((col('date')/100) % 100).cast('Integer')).withColumn('day', (col('date')% 100).cast('Integer'))
    
    weather_data = weather_data.select(
        'station_id',
        'date',
        'element',
        'data_value',
        'obs_time',
        'year',
        'month',
        'day'
    )
    
    weather_data = weather_data.join(ids_df, 'station_id', how='left')
    weather_data = weather_data.cache()
    
    
    #Show number of data for each city
    count_data = weather_data.groupBy('city').agg({'city':'count'})
    # count_data.show()
    
    
    nyc = weather_data.filter((weather_data['city'] == "New York"))
    la = weather_data.filter((weather_data['city'] == "Los Angeles"))
    boston = weather_data.filter((weather_data['city'] == "Boston"))
    chicago = weather_data.filter((weather_data['city'] == "Chicago"))
    seattle = weather_data.filter((weather_data['city'] == "Seattle"))
    atlanta = weather_data.filter((weather_data['city'] == "Atlanta"))
    sf = weather_data.filter((weather_data['city'] == "San Francisco"))
    toronto = weather_data.filter((weather_data['city'] == "Toronto"))
    vancouver = weather_data.filter((weather_data['city'] == "Vancouver"))
    calgary = weather_data.filter((weather_data['city'] == "Calgary"))
    montreal = weather_data.filter((weather_data['city'] == "Montreal"))

    nyc.write.json(output+"nyc", compression='gzip', mode='overwrite')
    la.write.json(output+"la", compression='gzip', mode='overwrite')
    boston.write.json(output+"boston", compression='gzip', mode='overwrite')
    chicago.write.json(output+"chicago", compression='gzip', mode='overwrite')
    seattle.write.json(output+"seattle", compression='gzip', mode='overwrite')
    atlanta.write.json(output+"atlanta", compression='gzip', mode='overwrite')
    sf.write.json(output+"sf", compression='gzip', mode='overwrite')
    toronto.write.json(output+"toronto", compression='gzip', mode='overwrite')
    vancouver.write.json(output+"vancouver", compression='gzip', mode='overwrite')
    calgary.write.json(output+"calgary", compression='gzip', mode='overwrite')
    montreal.write.json(output+"montreal", compression='gzip', mode='overwrite')

    
main()