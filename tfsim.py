# Databricks notebook source
!pip install tensorflow_similarity

# COMMAND ----------

import tensorflow_similarity as tfsim
import os
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
import tensorflow as tf
import pandas as pd
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

# COMMAND ----------

BATCH_SIZE = 32
NUM_EPOCHS = 20
SHAPE = (1000,)
TRAINING_SAMPLE_SIZE = 300000

# COMMAND ----------

tfsim.utils.tf_cap_memory()

# COMMAND ----------

def get_model():
    inputs = tf.keras.layers.Input(shape=SHAPE)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
    x = tf.keras.Sequential([
    tf.keras.layers.Dense(units=500, activation='relu')
    ])(x)
#     x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
#     x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
#     x = tf.keras.layers.MaxPool2D()(x)
#     x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
#     x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    # smaller embeddings will have faster lookup times while a larger embedding will improve the accuracy up to a point.
    outputs = tfsim.layers.MetricEmbedding(64)(x)
    return tfsim.models.SimilarityModel(inputs, outputs)

model = get_model()
model.summary()

# COMMAND ----------

def get_compiled_model(lr=0.000005, distance = "cosine"):  # @param {type:"number"}
    model = get_model()
    loss = tfsim.losses.MultiSimilarityLoss(distance=distance)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    return model

# COMMAND ----------

# MAGIC %md ### using Petastorm
# MAGIC [source](https://docs.databricks.com/_static/notebooks/deep-learning/petastorm-spark-converter-tensorflow.html)

# COMMAND ----------

df = spark.read.table("ceehacks_ecg_samples")
df_train, df_val, df_test = df.limit(TRAINING_SAMPLE_SIZE).randomSplit([0.8, 0.1, 0.1], seed=42)

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)
print(f"train: {len(converter_train)}, val: {len(converter_val)}")

# COMMAND ----------

df.count()

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

df_train.groupby(col('id')).count().display()

# COMMAND ----------

# def transform_row(pd_batch):
#     pd_batch['features'] = pd_batch['slice']
#     pd_batch['label_index'] = pd_batch['id']
#     pd_batch = pd_batch.drop(labels=['slice', 'id'], axis=1)
#     return pd_batch

# TODO additional transformations can be added here!

transform_spec_fn = TransformSpec(
#     transform_row,
    edit_fields=[('slice', np.float32, SHAPE, False), ('id', np.int32, (), False)], 
    selected_fields=['slice', 'id']
)

# COMMAND ----------

import mlflow

# COMMAND ----------

def train_and_evaluate():
    model = get_compiled_model(lr=0.001)
    with mlflow.start_run(run_name="similarity_model_training") as run:
        with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, batch_size=BATCH_SIZE) as train_dataset, \
             converter_val.make_tf_dataset(transform_spec=transform_spec_fn, batch_size=BATCH_SIZE) as val_dataset:

            # tf.keras only accept tuples, not namedtuples
            train_dataset = train_dataset.map(lambda x: (x.slice, x.id))
            steps_per_epoch = len(converter_train) // BATCH_SIZE

            val_dataset = val_dataset.map(lambda x: (x.slice, x.id))
            validation_steps = max(1, len(converter_val) // BATCH_SIZE)

            print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")
        # with mlflow.start_run(run_name="similarity_model_training") as run:
            hist = model.fit(train_dataset, 
                             steps_per_epoch=steps_per_epoch,
                             epochs=NUM_EPOCHS,
                             validation_data=val_dataset,
                             validation_steps=validation_steps,
                             verbose=2)
            mlflow.keras.log_model(
                 model,
              artifact_path="similarity_model",
              registered_model_name="similarity_model"
              )
#     return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1] 
    return hist.history
  
history = train_and_evaluate()
history

# COMMAND ----------

model_name = "similarity_model"
stage = 'Production'

inference_model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{stage}")

# COMMAND ----------

plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.legend(["loss", "val_loss"])
# plt.title(f"Loss: {loss.name} - LR: {LR}")
plt.show()

# COMMAND ----------

df_test_pandas = df_test.toPandas()
df_train_pandas = df_train.limit(5000).toPandas()

# COMMAND ----------

rv = []
for image_batch in np.array_split(df_test_pandas.slice, BATCH_SIZE):
    images = np.vstack(image_batch)
    dataset = tf.data.Dataset.from_tensor_slices(images).batch(BATCH_SIZE)
    preds = inference_model.predict(dataset)
    rv.extend(preds)

# COMMAND ----------

df_test_pandas['embeddings'] = rv

# COMMAND ----------

import sklearn

# COMMAND ----------

from sklearn.decomposition import PCA

# COMMAND ----------

df_test_pandas.embeddings

# COMMAND ----------

arr = [df_test_pandas.loc[i].embeddings.tolist() for i in range(len(df_test_pandas))]

# COMMAND ----------

pca = PCA(n_components=2)
pca.fit(arr)

# COMMAND ----------

transformed = pca.transform(arr)

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

X = [t[0] for t in transformed]
Y = [t[1] for t in transformed]

# COMMAND ----------

df_test_pandas.groupby('id').count()

# COMMAND ----------

x_n = X[0:10] #+ X[2500:2600]
y_n = Y[0:10] #+ Y[2500:2600]

x_n_n = X[300:310]
y_n_n = Y[300:310]

#x_n_n_n = X[3000:3100]
#y_n_n_n = Y[3000:3100]

# COMMAND ----------

plt.plot(df_test_pandas.iloc[4].slice)

# COMMAND ----------

plt.plot(df_test_pandas.iloc[300].slice)

# COMMAND ----------

plt.scatter(x_n, y_n)
plt.scatter(x_n_n, y_n_n, color='red')
#plt.scatter(x_n_n_n, y_n_n_n)

# COMMAND ----------

pca.explained_variance_ratio_

# COMMAND ----------


