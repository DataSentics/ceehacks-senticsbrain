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

plt.rcParams["figure.figsize"] = (20,13)

# COMMAND ----------

BATCH_SIZE = 64
NUM_EPOCHS = 20
SHAPE = (2000,)
TRAINING_SAMPLE_SIZE = 25000

# COMMAND ----------

df = spark.read.table("ceehacks_ecg_samples_slice_2000_n_samples_500")
df_train, df_val, df_test = df.orderBy(F.rand()).limit(TRAINING_SAMPLE_SIZE).randomSplit([0.7, 0.2, 0.1], seed=12)

# COMMAND ----------

df_train_pandas = df_train.toPandas()

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
    loss = tfsim.losses.TripletLoss(distance=distance)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    return model

# COMMAND ----------

sampler = tfsim.samplers.MultiShotMemorySampler(df_train_pandas['slice'], df_train_pandas['id'], classes_per_batch=10, examples_per_class_per_batch=4)

# COMMAND ----------

model = get_compiled_model(lr=0.0001)
with mlflow.start_run(run_name="similarity_model_training") as run:
    hist = model.fit(sampler, 
                     steps_per_epoch=1000,
                     epochs=40,
                     verbose=2,)
    mlflow.keras.log_model(
         model,
      artifact_path="similarity_model",
      registered_model_name="similarity_model"
      )

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

# COMMAND ----------

def visualize_pca_for_patients(df, n_patients = 3, n_samples_per_patient = 100):
    arr = [df.iloc[i].embeddings.tolist() for i in range(len(df))]
    pca = PCA(n_components=2)
    pca.fit(arr)
    print(f'Pca explained ration is {pca.explained_variance_ratio_}')
    df = df.sort_values('id')
    df = df.groupby('id').sample(frac=1)
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
df_pcad = visualize_pca_for_patients(df_train_pandas, n_patients=20, n_samples_per_patient=40)

# COMMAND ----------

plt.rcParams["figure.figsize"] = (13,8)

# COMMAND ----------

df_single = df_pcad[df_pcad.id == '24186521']
fig, ax = plt.subplots()
ax.plot(df_single.pca_x, df_single.pca_y, marker='o', linestyle='', ms=5, label=df_single.index)
centroid_x = df_single.pca_x.mean()
centroid_y =  df_single.pca_y.mean()
ax.plot(centroid_x, centroid_y, marker='o', linestyle='', ms=15, label=df_single.index)
ax.plot()
for idx, txt in enumerate(df_single.index):
    ax.annotate(txt, (df_single.loc[txt].pca_x, df_single.loc[txt].pca_y))

# COMMAND ----------

def plot_normal_and_outliers(df, id):
    def eukleid(x1, x2, y1, y2):
        return ((x1-x2)**2 + (y1-y2)**2)**(1/2)
    df_single = df[df['id'] == id]
    centroid_x = df_single.pca_x.mean()
    centroid_y = df_single.pca_y.mean()
    df_single['distance'] = df_single.pcad.apply(lambda x: eukleid(x[0], centroid_x, x[1], centroid_y))
    df_single = df_single.sort_values('distance')
    normal = df_single.head(4)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(df_single.iloc[0].slice)
    axs[0, 0].yaxis.grid(color='gray')
    axs[0, 0].xaxis.grid(color='gray')
    axs[0, 1].plot(df_single.iloc[1].slice)
    axs[0, 1].yaxis.grid(color='gray')
    axs[0, 1].xaxis.grid(color='gray')
    axs[1, 0].plot(df_single.iloc[2].slice)
    axs[1, 0].yaxis.grid(color='gray')
    axs[1, 0].xaxis.grid(color='gray')
    axs[1, 1].plot(df_single.iloc[3].slice)
    axs[1, 1].yaxis.grid(color='gray')
    axs[1, 1].xaxis.grid(color='gray')
    fig.suptitle('patient normal ECG')
    plt.show()
        
    mx = df_single.distance.max()
    m = df_single.distance.mean()
    if mx > 3*m:
        fig, axs = plt.subplots(2)
        axs[0].plot(df_single.iloc[-1].slice, color='red')
        axs[0].yaxis.grid(color='gray')
        axs[0].xaxis.grid(color='gray')
        axs[1].plot(df_single.iloc[-2].slice, color='red')
        axs[1].yaxis.grid(color='gray')
        axs[1].xaxis.grid(color='gray')
        fig.suptitle('Potential outliers')
        plt.show()
    else:
        print('This patient does not have any anomalies.')
    


# COMMAND ----------

plt.rcParams["figure.figsize"] = (10,5)
for id in df_pcad.id.unique()[:10]:
    plot_normal_and_outliers(df_pcad, id)

# COMMAND ----------

plt.plot(df_single.loc[2159].slice)

# COMMAND ----------

plt.plot(df_single.loc[2190].slice)

# COMMAND ----------


