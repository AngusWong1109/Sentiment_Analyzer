import sys
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import col, count, to_date, dayofmonth, month, year, when

spark = SparkSession.builder.appName('GHCN data').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

output = 'weather_output/weather-'

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
    "../dataset/2019.csv.gz",
    "../dataset/2020.csv.gz",
    "../dataset/2021.csv.gz",
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
    'CA001108380': 'Vancouver',
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
    'CA001108380',
    'CA003031092',
    'CA007025251',
]

elements = [
    'PRCP',
    'SNOW',
    'SNWD',
    'TMAX',
    'TMIN',
]

cities = []

weather_schema = types.StructType([
    types.StructField('station_id', types.StringType()),
    types.StructField('date', types.IntegerType()),
    types.StructField('element', types.StringType()),
    types.StructField('data_value', types.FloatType()),
    types.StructField('m_flag', types.StringType()),
    types.StructField('q_flag', types.StringType()),
    types.StructField('s_flag', types.StringType()),
    types.StructField('obs_time', types.IntegerType()),
])

def main():
    weather_data = spark.read.csv(weather_file_path_local, schema=weather_schema)
    ids_df = spark.createDataFrame(list(ids.items()), ["station_id", "city"])
    weather_data = weather_data.withColumn('date', to_date(weather_data.date, 'yyyyMMdd'))
    weather_data.drop('date')
    weather_data = weather_data.filter(
        (weather_data['station_id'].isin(stations)) &
        (weather_data['data_value'] != 9999) &
        (weather_data['element'].isin(elements))
    )
    
    weather_data = weather_data.withColumn('year', year(weather_data.date)).withColumn('month', month(weather_data.date)).withColumn('day', dayofmonth(weather_data.date))
    
    weather_data = weather_data.select(
        'station_id',
        'date',
        'element',
        'data_value',
        'year',
        'month',
        'day'
    )
    
    weather_data = weather_data.join(ids_df, 'station_id', how='left')
    weather_data = weather_data.cache()
    
    #Show number of data for each city
    # count_data = weather_data.groupBy('city').agg({'city':'count'})
    # weather_data.show()
    
    weather_data = weather_data.groupBy(['station_id', 'date', 'year', 'month', 'day', 'city']).pivot('element', elements).sum('data_value')
    weather_data = weather_data.withColumn('TMAX', (weather_data.TMAX / 10)).withColumn('TMIN', (weather_data.TMIN / 10))
    weather_data.drop('TMAX')
    weather_data.drop('TMIN')
    weather_data = weather_data.withColumn('TAVG', (weather_data.TMAX + weather_data.TMIN)/2)
    weather_data = weather_data.withColumn('T_label', when(weather_data.TAVG > 13, "hot").otherwise("cold"))
    
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
    
    nyc.write.csv(output + 'nyc', mode='overwrite', compression='gzip')
    la.write.csv(output + 'la', mode='overwrite', compression='gzip')
    boston.write.csv(output + 'boston', mode='overwrite', compression='gzip')
    chicago.write.csv(output + 'chicago', mode='overwrite', compression='gzip')
    seattle.write.csv(output + 'seattle', mode='overwrite', compression='gzip')
    atlanta.write.csv(output + 'atlanta', mode='overwrite', compression='gzip')
    sf.write.csv(output + 'sf', mode='overwrite', compression='gzip')
    toronto.write.csv(output + 'toronto', mode='overwrite', compression='gzip')
    vancouver.write.csv(output + 'vancouver', mode='overwrite', compression='gzip')
    calgary.write.csv(output + 'calgary', mode='overwrite', compression='gzip')
    montreal.write.csv(output + 'montreal', mode='overwrite', compression='gzip')
    
    """ 
    cities = [nyc, la, boston, chicago, seattle, atlanta, sf, toronto, vancouver, calgary, montreal]
    
    for city in cities:
        pdCity = city.toPandas()
        plt.plot(pdCity['date'], pdCity['TMIN'], 'r.')
        plt.plot(pdCity['date'], pdCity['TMAX'] , 'b.')
        plt.legend(['TMIN', 'TMAX'])
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.show() 
    """
    
main()