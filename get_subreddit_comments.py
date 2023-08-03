import sys
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('reddit extracter').getOrCreate()

reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/'
# reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
reddit_comments_path_list = [
    '/courses/datasets/reddit_comments_repartitioned/year=2015/*/*.json.gz',
    '/courses/datasets/reddit_comments_repartitioned/year=2016/*/*.json.gz',
    '/courses/datasets/reddit_comments_repartitioned/year=2017/*/*.json.gz',
    '/courses/datasets/reddit_comments_repartitioned/year=2018/*/*.json.gz',
    '/courses/datasets/reddit_comments_repartitioned/year=2019/*/*.json.gz',
    '/courses/datasets/reddit_comments_repartitioned/year=2020/*/*.json.gz',
    '/courses/datasets/reddit_comments_repartitioned/year=2021/*/*.json.gz',
]

output = 'output/reddit-subset'

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

def main():
    reddit_comments = spark.read.json(reddit_comments_path_list, schema=comments_schema)

    subs = ['vancouver', 'toronto','Calgary', 'montreal', 'nyc', 'LosAngeles', 'boston', 'chicago', 'Seattle','Atlanta','sanfrancisco']

    select_reddit_comments = reddit_comments.select(reddit_comments['subreddit'], reddit_comments['created_utc'], reddit_comments['body'])

    # select_reddit_comments.cache()
    select_reddit_comments.where(select_reddit_comments['subreddit'].isin(subs)) \
        .write.json(output + '/comments', mode='overwrite', compression='gzip')

main()