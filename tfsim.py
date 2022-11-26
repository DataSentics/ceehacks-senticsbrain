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
NUM_EPOCHS = 10
SHAPE = (1000,)
TRAINING_SAMPLE_SIZE = 1000

# COMMAND ----------

tfsim.utils.tf_cap_memory()

# COMMAND ----------

def get_model():
    inputs = tf.keras.layers.Input(shape=SHAPE)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
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

# df = pd.DataFrame(data={'ecg': [np.random.random((100)) for _ in range(10000)],
#                         'labels': [np.random.randint(100) for _ in range(10000)]})

# COMMAND ----------

# dataset = tf.data.Dataset.from_tensor_slices((list(df['ecg'].values), df['labels'].values)).batch(32)

# COMMAND ----------

# dataset = tf.data.Dataset.from_tensor_slices((list(df['ecg'].values), df['labels'].values))

# COMMAND ----------

# test_dataset = dataset.take(1000).batch(32)
# train_dataset = dataset.skip(1000).batch(32)

# COMMAND ----------

# EPOCHS = 10  # @param {type:"integer"}
# history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)

# COMMAND ----------

# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.legend(["loss", "val_loss"])
# plt.title(f"Loss: {loss.name} - LR: {LR}")
# plt.show()

# COMMAND ----------

# dataset = tf.data.Dataset.from_tensor_slices((list(df['ecg'].values), df['labels'].values)).batch(32)

# COMMAND ----------

# model.calibrate()

# COMMAND ----------

# model.lookup(dataset)

# COMMAND ----------

# dataset.unbatch().split()

# COMMAND ----------




# COMMAND ----------

# MAGIC %md ### using Petastorm
# MAGIC [source](https://docs.databricks.com/_static/notebooks/deep-learning/petastorm-spark-converter-tensorflow.html)

# COMMAND ----------

df = spark.read.table("ceehacks_ecg_samples")
df_train, df_val = df.limit(TRAINING_SAMPLE_SIZE).randomSplit([0.9, 0.1], seed=42)

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)
print(f"train: {len(converter_train)}, val: {len(converter_val)}")

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

def train_and_evaluate():
    model = get_compiled_model(lr=0.001)
    with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, batch_size=BATCH_SIZE) as train_dataset, \
         converter_val.make_tf_dataset(transform_spec=transform_spec_fn, batch_size=BATCH_SIZE) as val_dataset:
        
        # tf.keras only accept tuples, not namedtuples
        train_dataset = train_dataset.map(lambda x: (x.slice, x.id))
        steps_per_epoch = len(converter_train) // BATCH_SIZE

        val_dataset = val_dataset.map(lambda x: (x.slice, x.id))
        validation_steps = max(1, len(converter_val) // BATCH_SIZE)

        print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

        hist = model.fit(train_dataset, 
                         steps_per_epoch=steps_per_epoch,
                         epochs=NUM_EPOCHS,
                         validation_data=val_dataset,
                         validation_steps=validation_steps,
                         verbose=2)
#     return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1] 
    return hist.history
  
history = train_and_evaluate()
history

# COMMAND ----------


