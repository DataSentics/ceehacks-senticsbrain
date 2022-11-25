# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %md ### alivecors data

# COMMAND ----------

# MAGIC %fs ls dbfs:/ehh_hackathon_alivecors/AC116529

# COMMAND ----------

# MAGIC %md #### labsALL data

# COMMAND ----------

! apt-get install --upgrade p7zip-full

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /dbfs && curl -sS https://owncloud.ikem.cz/index.php/s/Gd3Km5g0eNY4Kwr/download > labsall.7z && 7z e labsall.7z

# COMMAND ----------

(
    spark.read.option("header",True)
    .option("delimiter", ";").csv('dbfs:/LabsALL\ IOL.csv')
     .write.format('delta').mode("overwrite").saveAsTable('ceehacks_labs_all')
)

# COMMAND ----------

# MAGIC %sql OPTIMIZE ceehacks_labs_all

# COMMAND ----------

labsall = spark.read.table('ceehacks_labs_all')

# COMMAND ----------

display(labsall)

# COMMAND ----------

display(labsall.where(F.col('Patient') == '10164861')) # single patient?

# COMMAND ----------

# MAGIC %md #### hackaton dataset

# COMMAND ----------

! find /dbfs/ehh_hack/ -name "2021-03-19T02_32_35.674Z.txt"

# COMMAND ----------

! ! find /dbfs/ehh_hack/ -type f -name "2022-11-09*"

# COMMAND ----------

! find 

# COMMAND ----------

! find /dbfs/ehh_hack/ -type f -printf "%f\n" | sort | uniq -c # count of all the files in the folder

# COMMAND ----------

# MAGIC %fs ls dbfs:/ehh_hack/10164861

# COMMAND ----------

spark.read.option("recursiveFileLookup","true").json('dbfs:/ehh_hack/')

# COMMAND ----------

patient_data = (
    spark.read.format("json").load('dbfs:/ehh_hack/*') # read all as json
    .withColumn('path', F.input_file_name())
)
# display(steps)

# COMMAND ----------

df_cleaned = (
    patient_data
    .withColumn('id', F.element_at(F.split(F.col('path'), '/'), -2)) # extract patient ID
    .withColumn('dataset', F.element_at(F.split(F.element_at(F.split(F.col('path'), '/'), -1), '\.'),1)) # extract dataset from the file type 
    .withColumn('date', # either date or datetime will be present - combine those columns to get either > cast to date
        F.when(F.col('datetime').isNotNull(), F.to_date(F.col('datetime')))
        .when(F.col('date').isNotNull(), F.to_date(F.col('date'), 'yyyy-MM-dd'))
    ) 
    .where(F.col('_corrupt_record').isNull()) # ignore corrupted data (dg.txt)
)

# check that all the corrupted rows were dg > OK
# display( 
#     df_cleaned.where((F.col('_corrupt_record').isNotNull()) & (F.col('dataset') != 'dg'))
# )

# COMMAND ----------

df_rest = (
    df_cleaned.where(F.col('dataset') != 'bp') # all except bp data
    .groupBy('id', 'date')
    .pivot("dataset")
    .agg(F.element_at(F.collect_list('v'),-1).alias('v')) # take always only last element if there are multiple events in a day
)

bp_cleaned = (
    df_cleaned.where(F.col('dataset') == 'bp') # bp data only
    .groupBy('id', 'date')
    .pivot('dataset')
    .agg(F.element_at(F.collect_list('sys'),-1).alias('sys'), F.element_at(F.collect_list('dia'),-1).alias('dia')) # take always only last element if there are multiple events in a day
)

df_complete = df_rest.join(bp_cleaned, ['id', 'date'], how='outer') # join the dataset
# display(df_complete)
# display(df_rest.join())
# display(df_complete.groupBy('id', 'datetime').count().where(F.col('count') > 1)) # check that we do not have any id+datetime duplicates
# display(df_complete.where( (F.size('energy') > 1) | (F.size('exercise') > 1) | (F.size('hrresting') > 1) | (F.size('hrwalking') > 1) | (F.size('waist') > 1) | (F.size('weight') > 1) )) # check that we do not have any arrays with multiple values for single id+datetime

# COMMAND ----------

# display(df_complete)

# COMMAND ----------

ecg = (
    spark.read.format("json").load('dbfs:/ehh_hack/*/ecg/*') # list all ecg records
    .withColumn('path', F.input_file_name())
    .withColumn('id', F.element_at(F.split(F.col('path'), '/'), -3)) # extract patient ID from folder
    .withColumn('date', F.to_date(F.element_at(F.split(F.element_at(F.split(F.col('path'), '/'), -1), 'T'), 1))) # extract dataset from the file type
    .groupBy('id', 'date')
    .agg(F.element_at(F.collect_list('samples'),-1).alias('ecg'), # take last measurement of the day
         F.element_at(F.collect_list('samplingFrequency'),-1).alias('samplingFrequency'),
         F.element_at(F.collect_list('flags'),-1).alias('flags')
    ) 
)

# COMMAND ----------

# write out dataset
(
    df_complete.join(ecg, ['id', 'date'], how='outer')
    .write.format('delta').mode("overwrite").saveAsTable('ceehacks_patientdata')
)

# COMMAND ----------

# MAGIC %sql OPTIMIZE ceehacks_patientdata

# COMMAND ----------

# MAGIC %sql SELECT * FROM ceehacks_patientdata

# COMMAND ----------

# MAGIC %sql SELECT count(distinct(id)) as patients_count FROM ceehacks_patientdata

# COMMAND ----------

# MAGIC %sql SELECT id,date,count(*) as c FROM ceehacks_patientdata GROUP BY id, date HAVING c > 1 -- check for duplicate id+date, should be empty

# COMMAND ----------

# MAGIC %sql ANALYZE TABLE ceehacks_patientdata COMPUTE STATISTICS NOSCAN

# COMMAND ----------


