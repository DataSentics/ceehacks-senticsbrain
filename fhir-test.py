# Databricks notebook source
! pip install fhirpy fhir.resources

# COMMAND ----------

from fhirpy import SyncFHIRClient
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.humanname import HumanName
from fhir.resources.contactpoint import ContactPoint
import json

# COMMAND ----------

FHIR_SERVER_URL = 'https://fhir.6cz61p53wx4b.static-test-account.isccloud.io'
FHIR_API_KEY = 'BlVIB4tZJq65I4OpTJadS8SqrC513vaXaxLOjU7T'

client = SyncFHIRClient(url=FHIR_SERVER_URL, extra_headers={"x-api-key": FHIR_API_KEY})

# COMMAND ----------

patients = client.resources('Patient').fetch()
patients

# COMMAND ----------

p = client.resources('Patient').fetch()[0].serialize()

Patient.parse_obj(p) # parse the object

# COMMAND ----------

#Get our patient resources in which we will be able to fecth and search
patients_resources = client.resources('Patient')

#Part 2----------------------------------------------------------------------------------------------------------------------------------------------------
#We want to create a patient and save it into our server

#Create a new patient using fhir.resources
patient0 = Patient()

patien

# COMMAND ----------

client.resources('Device').fetch_all().pop()

# COMMAND ----------


