# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A class to manage updating the status of an API request / Azure Batch Job using
the Cosmos DB table "batch_api_jobs".
"""

import logging
import os
import unittest
import uuid
from typing import Union, Optional

from azure.cosmos.cosmos_client import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from server_api_config import API_INSTANCE_NAME, COSMOS_ENDPOINT, COSMOS_WRITE_KEY
from server_utils import get_utc_time


log = logging.getLogger(os.environ['FLASK_APP'])


class JobStatusTable:
    """
    A wrapper around the Cosmos DB client. Each item in the table "batch_api_jobs" represents
    a request/Batch Job, and should have the following fields:
        - id: this is the job_id
        - api_instance
        - status
        - last_updated
        - call_params: the dict representing the body of the POST request from the user
    The 'status' field is a dict with the following fields:
        - request_status
        - message
        - num_tasks  (present after Batch Job created)
        - num_images  (present after Batch Job created)
    """
    # a job moves from created to running/problem after the Batch Job has been submitted
    allowed_statuses = ['created', # with messages such as  
                        # - Request received from React web. Pending upload image directory to Blob container
                        # TODO: other statuses generating from React Web workflow

                        # - Request received from API. Listing images next...
                        'submitting_job', # with messages such as 
                        # - {num_images} images listed; submitting the job...
                        'running', # with messages such as
                        # - Submitted {num_images} images to cluster in {num_tasks} shards. 
                        # - Check number {num_checks}, {num_tasks_succeeded} out of {num_tasks} shards have completed, successfully, {num_tasks_failed} shards have failed. 
                        'failed', # with messages such as
                        # - The number of images ({num_images}) requested for processing exceeds the maximum accepted {api_config.MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB} in one call
                        # - Error occurred while preparing the Batch job: {e}
                        'problem', # with messages such as
                        # - Please contact us. Error occurred while submitting the Batch job: {e}
                        # - Error occurred while starting the monitoring thread: {e}
                        # - Job unfinished after {num_checks} x {api_config.MONITOR_PERIOD_MINUTES} minutes, please contact us to retrieve the results. Number of succeeded shards: {num_tasks_succeeded}
                        # - Please contact us to retrieve the results. Error occurred while aggregating results: {e}
                        'completed', # with messages such as
                        # - 0 images found in provided list of images
                        # - Zero images found in container or in provided list of images
                        # - {
                        #     'num_failed_shards': num_tasks_failed,
                        #     'output_file_urls': {
                        #         'detections': output_sas_url # this is the downloadable JSON result
                        #     }
                        #   }
                        'canceled' # with message:
                        # - Request has been canceled by the user.
                        ]

    def __init__(self, api_instance=None):
        self.api_instance = api_instance if api_instance is not None else API_INSTANCE_NAME
        cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_WRITE_KEY)
        db_client = cosmos_client.get_database_client('camera-trap') #'camera-trap'
        self.db_jobs_client = db_client.get_container_client('batch_api_jobs')

    def create_job_status(self, job_id: str, status: Union[dict, str], call_params: dict) -> dict:
        assert 'request_status' in status and 'message' in status
        assert status['request_status'] in JobStatusTable.allowed_statuses

        # job_id should be unique across all instances, and is also the partition key
        cur_time = get_utc_time()
        print(self.db_jobs_client)
        item = {
            'id': job_id,
            'api_instance': self.api_instance,
            'status': status,
            'job_submission_time': cur_time,
            'last_updated': cur_time,
            'call_params': call_params
        }
        created_item = self.db_jobs_client.create_item(item)
        return created_item

    def update_job_status(self, job_id: str, status: Union[dict, str]) -> dict:
        assert 'request_status' in status and 'message' in status
        assert status['request_status'] in JobStatusTable.allowed_statuses

        # TODO do not read the entry first to get the call_params when the Cosmos SDK add a
        # patching functionality:
        # https://feedback.azure.com/forums/263030-azure-cosmos-db/suggestions/6693091-be-able-to-do-partial-updates-on-document
        item_old = self.read_job_status(job_id)
        if item_old is None:
            raise ValueError

        # need to retain other fields in 'status' to be able to restart monitoring thread
        if 'status' in item_old and isinstance(item_old['status'], dict):
            # retain existing fields; update as needed
            for k, v in item_old['status'].items():
                if k not in status:
                    status[k] = v
        item = {
            'id': job_id,
            'api_instance': self.api_instance,
            'status': status,
            'job_submission_time': item_old['job_submission_time'],
            'last_updated': get_utc_time(),
            'call_params': item_old['call_params']
        }
        replaced_item = self.db_jobs_client.replace_item(job_id, item)
        return replaced_item

    def read_job_status(self, job_id) -> Optional[dict]:
        """
        Read the status of the job from the Cosmos DB table of job status.
        Note that it does not check the actual status of the job on Batch, and just returns what
        the monitoring thread wrote to the database.
        job_id is also the partition key
        """
        try:
            read_item = self.db_jobs_client.read_item(job_id, partition_key=job_id)
            assert read_item['api_instance'] == self.api_instance, 'Job does not belong to this API instance'
        except CosmosResourceNotFoundError:
            return None  # job_id not a key. Task is not found
        except Exception as e:
            logging.error(f'server_job_status_table, read_job_status, exception: {e}')
            raise
        else:
            item = {k: v for k, v in read_item.items() if not k.startswith('_')}
            return item


class TestJobStatusTable(unittest.TestCase):
    api_instance = 'api_test'

    def test_insert(self):
        table = JobStatusTable(TestJobStatusTable.api_instance)
        status = {
            'request_status': 'running',
            'message': 'this is a test'
        }
        job_id = uuid.uuid4().hex
        item = table.create_job_status(job_id, status, {'container_sas': 'random_string'})
        self.assertTrue(job_id == item['id'], 'Expect job_id to be the id of the item')
        self.assertTrue(item['status']['request_status'] == 'running', 'Expect fields to be inserted correctly')

    def test_update_and_read(self):
        table = JobStatusTable(TestJobStatusTable.api_instance)
        status = {
            'request_status': 'running',
            'message': 'this is a test'
        }
        job_id = uuid.uuid4().hex
        res = table.create_job_status(job_id, status, {'container_sas': 'random_string'})

        status = {
            'request_status': 'completed',
            'message': 'this is a test again'
        }
        res = table.update_job_status(job_id, status)
        item_read = table.read_job_status(job_id)
        self.assertTrue(item_read['status']['request_status'] == 'completed', 'Expect field to have updated')

    def test_read_invalid_id(self):
        table = JobStatusTable(TestJobStatusTable.api_instance)
        job_id = uuid.uuid4().hex  # should not be in the database
        item_read = table.read_job_status(job_id)
        self.assertIsNone(item_read)


if __name__ == '__main__':
    unittest.main()
