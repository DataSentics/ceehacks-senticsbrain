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

# COMMAND ----------

BATCH_SIZE = 64
NUM_EPOCHS = 20
SHAPE = (2000,)
TRAINING_SAMPLE_SIZE = 200000

# COMMAND ----------

tfsim.utils.tf_cap_memory()

# COMMAND ----------

def get_model():
    inputs = tf.keras.layers.Input(shape=SHAPE)
    x = tf.keras.models.Sequential([ 
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Dense(128),
    ])(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tfsim.layers.MetricEmbedding(64)(x)
    return tfsim.models.SimilarityModel(inputs, outputs)

# COMMAND ----------

def get_model():
    inputs = tf.keras.layers.Input(shape=SHAPE)
    # x = tf.keras.layers.BatchNormalization()(inputs)
    # x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.Sequential([
    tf.keras.layers.Dense(units=512, activation='tanh'),    
    tf.keras.layers.Dense(units=256, activation='tanh')
    ])(inputs)
    
    x = tf.keras.layers.Flatten()(x)
    outputs = tfsim.layers.MetricEmbedding(128)(x)
    return tfsim.models.SimilarityModel(inputs, outputs)

model = get_model()
model.summary()

# COMMAND ----------

def get_compiled_model(lr=0.00005, distance = "cosine"):  # @param {type:"number"}
    model = get_model()
    loss = tfsim.losses.MultiSimilarityLoss(distance=distance)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss)
    return model

# COMMAND ----------

# MAGIC %md ### using Petastorm
# MAGIC [source](https://docs.databricks.com/_static/notebooks/deep-learning/petastorm-spark-converter-tensorflow.html)

# COMMAND ----------

df = spark.read.table("ceehacks_ecg_samples_slice_2000_n_samples_100")
df_train, df_val, df_test = df.limit(TRAINING_SAMPLE_SIZE).orderBy(F.rand()).randomSplit([0.7, 0.2, 0.1], seed=12)

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)
print(f"train: {len(converter_train)}, val: {len(converter_val)}")

# COMMAND ----------

transform_spec_fn = TransformSpec(
    edit_fields=[('slice', np.float32, SHAPE, False), ('id', np.int32, (), False)], 
    selected_fields=['slice', 'id']
)

# COMMAND ----------

df_train_pandas = df_train.limit(50000).toPandas()

# COMMAND ----------

df_train_pandas

# COMMAND ----------

sampler = tfsim.samplers.MultiShotMemorySampler(df_train_pandas['slice'], df_train_pandas['id'], classes_per_batch=8, examples_per_class_per_batch=4)

# COMMAND ----------

model = get_compiled_model(lr=0.005)
with mlflow.start_run(run_name="similarity_model_training") as run:
# with mlflow.start_run(run_name="similarity_model_training") as run:
    hist = model.fit(sampler, 
                     steps_per_epoch=1000,
                     epochs=NUM_EPOCHS,
                     verbose=2)
    mlflow.keras.log_model(
         model,
      artifact_path="similarity_model",
      registered_model_name="similarity_model"
      )

# COMMAND ----------

def train_and_evaluate():
    model = get_compiled_model(lr=0.0005)
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

plt.plot(hist.history["loss"])
plt.legend(["loss", "val_loss"])
# plt.title(f"Loss: {loss.name} - LR: {LR}")
plt.show()

# COMMAND ----------

# MAGIC %md ## Load model

# COMMAND ----------

model_name = "similarity_model"
stage = 'Production'

inference_model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{stage}")

# COMMAND ----------

df_test_pandas = df_test.toPandas()
df_train_pandas = df_train.limit(10000).toPandas()

# COMMAND ----------

inference_model = model

# COMMAND ----------

def append_emebddings_to_df(df):
    rv = []
    for image_batch in np.array_split(df.slice, BATCH_SIZE):
        images = np.vstack(image_batch)
        dataset = tf.data.Dataset.from_tensor_slices(images).batch(BATCH_SIZE)
        preds = inference_model.predict(dataset)
        rv.extend(preds)
    df['embeddings'] = rv
    return df

# COMMAND ----------

df_train_pandas = append_emebddings_to_df(df_train_pandas)
df_test_pandas = append_emebddings_to_df(df_test_pandas)

# COMMAND ----------

def visualize_pca_for_patients(df, n_patients = 3, n_samples_per_patient = 100):
    # n_samples_per_patient = 100
    arr = [df.loc[i].embeddings.tolist() for i in range(len(df))]
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


# COMMAND ----------

visualize_pca_for_patients(df_train_pandas, n_patients=2, n_samples_per_patient=10)

# COMMAND ----------


