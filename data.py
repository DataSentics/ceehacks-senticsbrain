# Databricks notebook source
df = spark.read.table('ceehacks_patientdata')

# COMMAND ----------

df.display()

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

with_ecg = df.filter(~F.col('ecg').isNull())

# COMMAND ----------

with_ecg.display()

# COMMAND ----------

with_ecg.select('id', 'ecg').withColumn('ecg_samples', F.slice(F.col('ecg'), 1, 10)).display()

# COMMAND ----------


