# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A module to hold the configurations specific to an instance of the API.
"""


#%% instance-specific API settings
# you likely need to modify these when deploying a new instance of the API
SENTINEL = 'awc'  # sentinel key in Azure App Configuration to check for changes

API_INSTANCE_NAME = 'malanet' # existing instance: cp (sponsorship), malanet (payg)
POOL_ID_SLOW = 'pool_v100_5tasks_3000'
POOL_ID_FAST = 'pool_a100_20tasks_3000' 
POOL_ID = POOL_ID_SLOW

MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB = 4 * 1000 * 1000  # inclusive

# Azure Batch for batch processing
BATCH_ACCOUNT_NAME = 'batchaccountname'
BATCH_ACCOUNT_URL = 'https://batchaccountname.australiaeast.batch.azure.com'

#%% general API settings
# API_PREFIX = '/v4/camera-trap/detection-batch'  # URL to root is http://127.0.0.1:5000/v4/camera-trap/detection-batch/

MONITOR_PERIOD_MINUTES = 5

# if this number of times the thread wakes up to check is exceeded, stop the monitoring thread
MAX_MONITOR_CYCLES = 7 * 24 * int(60 / MONITOR_PERIOD_MINUTES)  # 1 week

IMAGE_SUFFIXES_ACCEPTED = ('.jpg', '.jpeg', '.png')  # case-insensitive

OUTPUT_FORMAT_VERSION = '1.4'
NUM_IMAGES_PER_TASK = 3000 
OUTPUT_SAS_EXPIRATION_DAYS = 90 # 3 months

# quota of active Jobs in our Batch account, which all node pools i.e. API instances share;
# cannot accept job submissions if there are this many active Jobs already
MAX_BATCH_ACCOUNT_ACTIVE_JOBS = 100

SHARED_MEMORY_SIZE=10

#%% MegaDetector info
DETECTION_CONF_THRESHOLD = 0.1

# relative to the `megadetector_copies` folder in the container `models`
MD_VERSIONS_TO_REL_PATH = {
    # '4.1': 'megadetector_v4_1/md_v4.1.0.pb',
    '5a':'megadetector_v5/md_v5a.0.0.pt',
    '5b':'megadetector_v5/md_v5b.0.0.pt'
}
DEFAULT_MD_VERSION = '5a'
assert DEFAULT_MD_VERSION in MD_VERSIONS_TO_REL_PATH

DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'
}

#%% Classifier info
CLASSIFIER_CONF_THRESHOLD=0.3
CLASSIFIER_BATCH_SIZE=16
CLASSIFIER_N_WORKERS=2
CLASSIFIER_TTA=2
#%% Azure Batch settings
NUM_TASKS_PER_SUBMISSION = 20  # max for the Python SDK without extension is 100
NUM_TASKS_PER_RESUBMISSION = 5


#%% env variables for service credentials, and info related to these services

# Cosmos DB `batch-api-jobs` table for job status
COSMOS_ENDPOINT = 'Your Cosmos DB endpoint'
COSMOS_WRITE_KEY = 'Your Cosmos DB write key'
# this is the PRIMARY KEY (read-write keys)

# Service principal of this "application", authorized to use Azure Batch
APP_TENANT_ID = 'Your App Tenant ID' 
APP_CLIENT_ID = 'Your App Client ID'
APP_CLIENT_SECRET = 'Your App Client Secret'

# Blob storage account for storing Batch tasks' outputs and scoring script
STORAGE_ACCOUNT_NAME =  'Your Blob storage account name' 
STORAGE_ACCOUNT_KEY = 'Your Blob storage account key' # os.environ['STORAGE_ACCOUNT_KEY']

# Blob Containers
# |__ batch-api
# |   |__ api_<API_INSTANCE_NAME>
# |   |   |__ jobs_<job_id>
# |   |   |__ jobs_<job_id>
# |   |   |__ ...
# |   |__ scripts_v2
# |       |__ nvidia_gpu_drivers_startup.sh
# |       |__ score.py
# |
# |__ models
# |   |__ classifiers
# |   |   |__ classification_params.json
# |   |   |__ <weight>.pth
# |   |__ megadetector_copies
# |       |__ megadetector_v5
# |           |__ md_v5a.0.0.pt
# |           |__ md_v5b.0.0.pt
# |
# |__ test-centralised-upload
#     |__ <image_folder>
#     |__ <image_folder>

STORAGE_CONTAINER_API = 'batch-api'
STORAGE_CONTAINER_MODELS = 'models'
STORAGE_DETECTOR_DIR = 'megadetector_copies'
STORAGE_CLASSIFIER_DIR = 'classifiers'
STORAGE_CLASSIFIER_PARAMS = 'classification_params.json'

# Azure Container Registry for Docker image used by our Batch node pools
CONTAINER_IMAGE_NAME = 'imgclfctnreg.azurecr.io/pytorch:2.4.1-cuda12.1-cudnn9-runtime' # default docker image for the node pool