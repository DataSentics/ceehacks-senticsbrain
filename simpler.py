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
import mlflow
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
tfsim.utils.tf_cap_memory()

# COMMAND ----------

BATCH_SIZE = 64
NUM_EPOCHS = 20
SHAPE = (1000,)
TRAINING_SAMPLE_SIZE = 25000

# COMMAND ----------

# MAGIC %md ## Load data

# COMMAND ----------

df = spark.read.table("ceehacks_ecg_samples_slice_1000_n_samples_500")
df_train, df_val, df_test = df.orderBy(F.rand()).limit(TRAINING_SAMPLE_SIZE).randomSplit([0.7, 0.2, 0.1], seed=12)

# COMMAND ----------

df_train_pandas = df_train.toPandas()
# df_val_pandas = df_val.toPandas()

# COMMAND ----------

# MAGIC %md ## Get model

# COMMAND ----------

def get_model():
    inputs = tf.keras.layers.Input(shape=SHAPE)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dense(128)(x)
    # x = tf.keras.layers.Flatten()(x)
    outputs = tfsim.layers.MetricEmbedding(64)(x)
    return tfsim.models.SimilarityModel(inputs, outputs)

model = get_model()
model.summary()

# COMMAND ----------

def get_compiled_model(lr=0.00005, distance = "cosine"):  # @param {type:"number"}
    model = get_model()
    loss = tfsim.losses.MutiSimilarityLoss(distance=distance)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    return model

# COMMAND ----------

sampler = tfsim.samplers.MultiShotMemorySampler(df_train_pandas['slice'], df_train_pandas['id'], classes_per_batch=10, examples_per_class_per_batch=4)

# COMMAND ----------

df_train_pandas.groupby('id').count()

# COMMAND ----------

model = get_compiled_model(lr=0.0001)
with mlflow.start_run(run_name="similarity_model_training") as run:
    hist = model.fit(sampler, 
                     steps_per_epoch=1000,
                     epochs=20,
                     verbose=2,)
    mlflow.keras.log_model(
         model,
      artifact_path="similarity_model",
      registered_model_name="similarity_model"
      )

# COMMAND ----------

plt.plot(hist.history["loss"])
plt.legend(["loss", "val_loss"])
# plt.title(f"Loss: {loss.name} - LR: {LR}")
plt.show()

# COMMAND ----------

# MAGIC %md ## Get embeddings

# COMMAND ----------

def append_emebddings_to_df(df, model):
    rv = []
    for image_batch in np.array_split(df.slice, BATCH_SIZE):
        images = np.vstack(image_batch)
        dataset = tf.data.Dataset.from_tensor_slices(images).batch(BATCH_SIZE)
        preds = model.predict(dataset)
        rv.extend(preds)
    df['embeddings'] = rv
    return df

# COMMAND ----------

df_train_pandas = append_emebddings_to_df(df_train_pandas, model)
df_test_pandas = append_emebddings_to_df(df_val_pandas, model)

# COMMAND ----------

import umap

# COMMAND ----------

def visualize_pca_for_patients(df, n_patients = 3, n_samples_per_patient = 100):
    # n_samples_per_patient = 100
    arr = [df.iloc[i].embeddings.tolist() for i in range(len(df))]
    pca = PCA(n_components=2)
    pca.fit(arr)
    print(f'Pca explained ration is {pca.explained_variance_ratio_}')
    df = df.sort_values('id')
    df = df.groupby('id').sample(n_samples_per_patient)
    df['pcad'] = df.embeddings.apply(lambda x: pca.transform([x.tolist()])[0])
    df['pca_x'] = df['pcad'].apply(lambda x: x[0])
    df['pca_y'] = df['pcad'].apply(lambda x: x[1])
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    groups = df.groupby('id')
    i = 0
    for name, group in groups:
        ax.plot(group.pca_x, group.pca_y, marker='o', linestyle='', ms=5, label=name)
        i += 1
        if i >= n_patients:
            break
    ax.legend()

    plt.show()
    return df


# COMMAND ----------

plt.rcParams["figure.figsize"] = (20,13)
df_pcad = visualize_pca_for_patients(df_train_pandas, n_patients=25, n_samples_per_patient=25)

# COMMAND ----------

plt.rcParams["figure.figsize"] = (20,13)
df_single = df_pcad[df_pcad.id == '16512904']
fig, ax = plt.subplots()
ax.plot(df_single.pca_x, df_single.pca_y, marker='o', linestyle='', ms=5, label=df_single.index)
for idx, txt in enumerate(df_single.index):
    ax.annotate(txt, (df_single.loc[txt].pca_x, df_single.loc[txt].pca_y))

# COMMAND ----------

plt.rcParams["figure.figsize"] = (12,8)

# COMMAND ----------

plt.plot(df_single.loc[2292].slice)

# COMMAND ----------

plt.plot(df_single.loc[2259].slice)

# COMMAND ----------

new_row =df_single.loc[N, :]
record = ([0 for i in range(10)]) + list(df_single.loc[N].slice[10:200].tolist() + ([70000 for i in range(300)]) + (df_single.loc[N].slice[500:].tolist()))
new_row['slice'] = record

# COMMAND ----------

df_appended = df_single.append(new_row).reset_index(drop=True)

# COMMAND ----------

embeds = append_emebddings_to_df(pd.concat([df_appended, df_appended, df_appended, df_appended]).reset_index(), model)

# COMMAND ----------

embeds

# COMMAND ----------

embeds = embeds[-20:]

# COMMAND ----------

embeds

# COMMAND ----------

pcad = visualize_pca_for_patients(embeds, n_patients=1, n_samples_per_patient=20)

# COMMAND ----------

pcad

# COMMAND ----------

df_single = pcad
fig, ax = plt.subplots()
ax.plot(df_single.pca_x, df_single.pca_y, marker='o', linestyle='', ms=5, label=df_single.index)
for idx, txt in enumerate(df_single.index):
    ax.annotate(txt, (df_single.loc[txt].pca_x, df_single.loc[txt].pca_y))
plt.rcParams["figure.figsize"] = (20,13)
plt.show()

# COMMAND ----------

plt.plot(df_single.loc[66].slice)

# COMMAND ----------

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(df_single.loc[68].slice)
plt.show()
plt.plot(df_single.loc[72].slice)
plt.show()
plt.plot(df_single.loc[64].slice)

# COMMAND ----------



# COMMAND ----------


