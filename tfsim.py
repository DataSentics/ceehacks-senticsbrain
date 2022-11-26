# Databricks notebook source
# MAGIC %pip install tensorflow_similarity

# COMMAND ----------

import tensorflow_similarity as tfsim
import os
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
import tensorflow as tf

# COMMAND ----------

tfsim.utils.tf_cap_memory()

# COMMAND ----------

def get_model():
    inputs = tf.keras.layers.Input(shape=(100,))
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
    #x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    #x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    #x = tf.keras.layers.MaxPool2D()(x)
    #x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    #x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    #x = tf.keras.layers.Flatten()(x)
    # smaller embeddings will have faster lookup times while a larger embedding will improve the accuracy up to a point.
    outputs = tfsim.layers.MetricEmbedding(64)(x)
    return tfsim.models.SimilarityModel(inputs, outputs)


model = get_model()
model.summary()

# COMMAND ----------

distance = "cosine"
loss = tfsim.losses.MultiSimilarityLoss(distance=distance)

# COMMAND ----------

LR = 0.000005  # @param {type:"number"}
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=loss)

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.DataFrame(data={'ecg': [np.random.random((100)) for _ in range(10000)],
                        'labels': [np.random.randint(100) for _ in range(10000)]})

# COMMAND ----------

dataset = tf.data.Dataset.from_tensor_slices((list(df['ecg'].values), df['labels'].values)).batch(32)

# COMMAND ----------

dataset = tf.data.Dataset.from_tensor_slices((list(df['ecg'].values), df['labels'].values))

# COMMAND ----------

test_dataset = dataset.take(1000).batch(32)
train_dataset = dataset.skip(1000).batch(32)

# COMMAND ----------

EPOCHS = 10  # @param {type:"integer"}
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)

# COMMAND ----------

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.title(f"Loss: {loss.name} - LR: {LR}")
plt.show()

# COMMAND ----------

dataset = tf.data.Dataset.from_tensor_slices((list(df['ecg'].values), df['labels'].values)).batch(32)

# COMMAND ----------

model.calibrate()

# COMMAND ----------

model.lookup(dataset)

# COMMAND ----------

dataset.unbatch().split()

# COMMAND ----------


