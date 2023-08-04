# CMPT353Group14
library required: numpy, pandas, sys, scipy, matplotlib, sklearn, pyspark.sql, pyspark.ml, statsmodels, vaderSentiment

### Step to work on the **whole** dataset on cluster:
1. Execute get_subreddit_comments.py by code: `spark-submit get_subreddit_comments.py`
2. Execute get_weather_data.py by code: `spark-submit get_weather_data.py`
3. Execute setup_city_subreddits.py by code: `spark-submit setup_city_subreddits.py `
4. Run analysis:
    - Execute rain_analysis.py by code: `spark-submit rain_analysis.py whole`
    - Execute snow_analysis.py by code: `spark-submit snow_analysis.py whole`
    - Execute cold_analysis.py by code: `spark-submit cold_analysis.py whole`
    - Execute hot_analysis.py by code: `spark-submit hot_analysis.py whole`
    - Execute random_forest.py by code: `spark-submit random_forest.py whole`

### Step to work on the **sample** dataset only on local machine:
1. Make sure you have cloned or downloaded all the files from this repositary
2. Run analysis:
    - Execute rain_analysis.py by code: `spark-submit rain_analysis.py sample`
    - Execute snow_analysis.py by code: `spark-submit snow_analysis.py sample`
    - Execute cold_analysis.py by code: `spark-submit cold_analysis.py sample`
    - Execute hot_analysis.py by code: `spark-submit hot_analysis.py sample`
    - Execute random_forest.py by code: `spark-submit random_forest.py sample`

### File produced
1. allcities-scatter.png
2. allcities-hist.png
3. allcities-snow-scatter.png
4. allcities-snow-hist.png
5. cold.png
6. hot.png
