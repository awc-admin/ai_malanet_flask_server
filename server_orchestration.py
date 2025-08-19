# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Functions to submit images to the Azure Batch node pool for processing, monitor
the Job and fetch results when completed.
"""

import io
import json
import threading
import time
import logging
import os
import urllib.parse
from datetime import timedelta, datetime, timezone
from random import shuffle

import sas_blob_utils
import requests
from azure.storage.blob import ContainerClient, BlobSasPermissions, generate_blob_sas
from tqdm import tqdm

from server_utils import *
import server_api_config as api_config
from server_batch_job_manager import BatchJobManager
from server_job_status_table import JobStatusTable
from pathlib import Path
import collections
from typing import Any, Dict, List
from megadetector.postprocessing.classification_postprocessing import smooth_classification_results_sequence_level, ClassificationSmoothingOptions

def sequence_assignment(
    image_data: List[Dict[str, Any]], 
    time_gap: int = 3
) -> Dict[str, List[Dict[str, str]]]:
    """
    Assigns sequence IDs to camera trap images based on their location and timestamp.

    This function groups images by their parent directory, sorts them chronologically,
    and then segments them into sequences. A new sequence is started if the time
    difference between an image and its predecessor exceeds the specified time_gap.

    Args:
        image_data: A list of dictionaries, where each dictionary represents an
                    image and must contain 'file' and 'datetime' keys.
        time_gap: The maximum time in seconds allowed between consecutive
                  images in the same sequence. Defaults to 3.

    Returns:
        A dictionary with a single key 'images', containing a list of
        dictionaries, each with 'file_name' and its assigned 'seq_id'.
        
    Raises:
        ValueError: If an image dictionary is missing 'file' or 'datetime' keys,
                    or if the datetime string has an invalid format.
    """
    grouped_by_dir = collections.defaultdict(list)
    for image_info in image_data:
        try:
            filepath = Path(image_info['file'])
            parent_dir = str(filepath.parent)
            
            datetime_obj = datetime.strptime(
                image_info['datetime'], '%Y:%m:%d %H:%M:%S'
            )
            
            grouped_by_dir[parent_dir].append({
                'file': image_info['file'], 
                'datetime_obj': datetime_obj
            })
        except KeyError as e:
            raise KeyError(f"Input image dictionary is missing required key: {e}")
        except ValueError:
            raise ValueError(
                f"Invalid datetime format for file {image_info.get('file')}. "
                "Expected 'YYYY:MM:DD HH:MM:SS'."
            )

    final_results = []
    time_delta_gap = datetime.timedelta(seconds=time_gap)

    for parent_dir, images_in_group in grouped_by_dir.items():
        if not images_in_group:
            continue
            
        sorted_images = sorted(
            images_in_group, 
            key=lambda x: (x['file'],x['datetime_obj'])
        )


        sequence_counter = 1
        
        # The first image always starts the first sequence for its directory.
        first_image = sorted_images[0]
        seq_id = f"{parent_dir}/sequence_{sequence_counter}"
        final_results.append({'file_name': first_image['file'], 'seq_id': seq_id})

        for i in range(1, len(sorted_images)):
            current_image = sorted_images[i]
            previous_image = sorted_images[i - 1]

            # If the gap is too large, start a new sequence.
            if current_image['datetime_obj'] - previous_image['datetime_obj'] > time_delta_gap:
                sequence_counter += 1
            
            seq_id = f"{parent_dir}/sequence_{sequence_counter}"
            final_results.append({
                'file_name': current_image['file'], 
                'seq_id': seq_id
            })
            
    return {'images': final_results}

# Gunicorn logger handler will get attached if needed in server.py
log = logging.getLogger(os.environ['FLASK_APP'])
class ClassificationParams:
    def __init__(self):
        self.classifier_weight,self.classifier_label_names,self.classifier_parent_names,self.classifier_params_path,\
        self.classifier_conf_threshold,self.classifier_hitax_threshold,\
        self.smoothing_sequence_timegap,self.smoothing_min_dominant,self.smoothing_max_nondominant = None,None,None,None,\
            None,None,None,None,None

def upload_imgs_to_container(image_paths,job_id):
    # upload the image list (byte) to the container, which is also mounted on all nodes
    # all sharding and scoring use the uploaded list
    images_list_str_as_bytes = bytes(json.dumps(image_paths, ensure_ascii=False), encoding='utf-8')
    container_url = sas_blob_utils.build_azure_storage_uri(account=api_config.STORAGE_ACCOUNT_NAME,
                                                            # australianwildlifedia830
                                                            container=api_config.STORAGE_CONTAINER_API
                                                            # batch-api
                                                            )
    with ContainerClient.from_container_url(container_url,
                                            credential=api_config.STORAGE_ACCOUNT_KEY) as api_container_client:
        _ = api_container_client.upload_blob(
            name=f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/{job_id}_images.json',
            # api_cp/job_{job_id}/{job_id}_images.json
            data=images_list_str_as_bytes)

def get_classification_params_from_container(do_classification=False, hitax_type='off', do_smoothing=False):
    classifier_params = ClassificationParams()
    if not do_classification:
        return classifier_params
    
    # get classification params JSON
    container_url = sas_blob_utils.build_azure_storage_uri(account=api_config.STORAGE_ACCOUNT_NAME,
                                                            # australianwildlifedia830
                                                           container=api_config.STORAGE_CONTAINER_MODELS
                                                            # models
                                                            )
    with ContainerClient.from_container_url(container_url,
                                            credential=api_config.STORAGE_ACCOUNT_KEY) as container_client:
        json_param_name = api_config.STORAGE_CLASSIFIER_HITAX_PARAMS if hitax_type=='hitax classifier' else api_config.STORAGE_CLASSIFIER_PARAMS
        blob_client = container_client.get_blob_client(f'{api_config.STORAGE_CLASSIFIER_DIR}/{json_param_name}')

        if blob_client.exists():  
            """
            Structure of the JSON file:
            {
                "model_weight": "model_file_name", # just the file name
                "parent_names": ["parent1", "parent2", ...],
                "label_names": ["label1", "label2", ...],
                "parent2child": {
                    "parent1": ["child1", "child2"],
                    "parent2": ["child3", "child4"]
                }, # for rollup
                "child2parent": {
                    "child1": ["parent1"],
                    "child2": ["parent1"],
                    "child3": ["parent2"],
                    "child4": ["parent2"]
                }, # for hitax classifier
                "tfms_params": {
                    "size": 300,
                    "max_rotate": 10,
                    "max_zoom": 1.3,
                    "min_scale": 1.0,
                    "do_flip": true,
                    "max_lighting": 0.5,
                    "p_lighting": 0.75,
                    "item_tfms_size": 300
                    }
                # some classifier params
            }
            """
            stream = io.BytesIO()
            blob_client.download_blob().readinto(stream)
            stream.seek(0)
            classification_json = json.load(stream)
            classifier_params.classifier_model_weight = classification_json['model_weight']
            classifier_params.classifier_label_names = classification_json['label_names']
            classifier_params.classifier_conf_threshold = classification_json.get('classifier_conf_threshold', api_config.CLASSIFIER_CONF_THRESHOLD)
            classifier_params.classifier_params_path = (Path(api_config.STORAGE_CONTAINER_MODELS)/api_config.STORAGE_CLASSIFIER_DIR/json_param_name).as_posix()

            if hitax_type in api_config.HITAX_ALL_TYPES[1:] and 'parent_names' in classification_json:
                classifier_params.classifier_parent_names = classification_json['parent_names']
                classifier_params.classifier_hitax_threshold = classification_json.get('classifier_hitax_threshold', api_config.CLASSIFIER_HITAX_THRESHOLD)

            if do_smoothing:
                classifier_params.smoothing_sequence_timegap = int(classification_json.get('smoothing_sequence_timegap', api_config.SMOOTHING_SEQUENCE_TIMEGAP))
                classifier_params.smoothing_min_dominant = int(classification_json.get('smoothing_min_dominant', api_config.SMOOTHING_MIN_DOMINANT))
                classifier_params.smoothing_max_nondominant = int(classification_json.get('smoothing_max_nondominant', api_config.SMOOTHING_MAX_NONDOMINANT))
    return classifier_params


def create_batch_job(job_id: str, body: dict):
    """
    This is the target to be run in a thread to submit a batch processing job and monitor progress
    """
    job_status_table = JobStatusTable()
    try:
        log.info(f'server_job, create_batch_job, job_id {job_id}, {body}')

        input_container_sas = body.get('input_container_sas', None)

        use_url = body.get('use_url', False)
        if use_url and isinstance(use_url, str):  # in case it is included but is intended to be False
            if use_url.lower() in ['false', 'f', 'no', 'n','0']:
                use_url = False
            else:
                use_url = True

        images_requested_json_sas = body.get('images_requested_json_sas', None)

        image_path_prefix = body.get('image_path_prefix', None)

        first_n = body.get('first_n', None)
        first_n = int(first_n) if first_n else None

        sample_n = body.get('sample_n', None)
        sample_n = int(sample_n) if sample_n else None

        model_version = body.get('model_version', '')
        if model_version == '':
            model_version = api_config.DEFAULT_MD_VERSION
        
        do_classification = body.get('classify', False) 
        if isinstance(do_classification, str): 
            do_classification = not do_classification.lower() in ['false', 'f', 'no', 'n', '0']

        # 'off', 'label rollup', 'hitax classifier'
        # for hitax, can output parent only, or parent-child mixed
        hitax_type = body.get('hitax_type', api_config.HITAX_ALL_TYPES[0])
        hitax_output = api_config.HITAX_OUTPUT_TYPES[0]
        if isinstance(hitax_type, str) and do_classification:
            hitax_type = hitax_type.lower()
            if hitax_type not in api_config.HITAX_ALL_TYPES:
                hitax_type = api_config.HITAX_ALL_TYPES[0]  # default
            elif hitax_type == 'hitax classifier':
                hitax_output = body.get('hitax_output', api_config.HITAX_OUTPUT_TYPES[0])
                if isinstance(hitax_output, str):
                    hitax_output = hitax_output.lower()
                    if hitax_output not in api_config.HITAX_OUTPUT_TYPES:
                        hitax_output = api_config.HITAX_OUTPUT_TYPES[0]  # default
                else:
                    hitax_output = api_config.HITAX_OUTPUT_TYPES[0]  # default, if not specified
        else:
            hitax_type = api_config.HITAX_ALL_TYPES[0] 

        do_smoothing = body.get('do_smoothing', False)
        if isinstance(do_smoothing, str): 
            do_smoothing = not do_smoothing.lower() in ['false', 'f', 'no', 'n', '0']

        # request_name and request_submission_timestamp are for appending to
        # output file names
        job_name = body.get('request_name', '')  # in earlier versions we used "request" to mean a "job"
        job_submission_timestamp = get_utc_time()

        # image_paths can be a list of strings (Azure blob names or public URLs)
        # or a list of length-2 lists where each is a [image_id, metadata] pair

        # Case 1: listing all images in the container
        # - not possible to have attached metadata if listing images in a blob
        if images_requested_json_sas is None:
            log.info('server_job, create_batch_job, listing all images to process.')

            # list all images to process
            image_paths = sas_blob_utils.list_blobs_in_container(
                container_uri=input_container_sas,
                blob_prefix=image_path_prefix,  # check will be case-sensitive
                blob_suffix=api_config.IMAGE_SUFFIXES_ACCEPTED,  # check will be case-insensitive
                limit=api_config.MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB + 1
                # + 1 so if the number of images listed > MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB
                # we will know and not proceed
            )
            print("Listed successful")

        # Case 2: user supplied a list of images to process; can include metadata
        else:
            log.info('server_job, create_batch_job, using provided list of images.')

            response = requests.get(images_requested_json_sas) # could be a file hosted anywhere
            image_paths = response.json()

            log.info('server_job, create_batch_job, length of image_paths provided by the user: {}'.format(len(image_paths)))
            if len(image_paths) == 0:
                job_status = get_job_status(
                    'completed', '0 images found in provided list of images.')
                job_status_table.update_job_status(job_id, job_status)
                return

            error, metadata_available = validate_provided_image_paths(image_paths)
            if error is not None:
                msg = 'image paths provided in the json are not valid: {}'.format(error)
                raise ValueError(msg)

            # filter down to those conforming to the provided prefix and accepted suffixes (image file types)
            valid_image_paths = []
            for p in image_paths:
                locator = p[0] if metadata_available else p

                # prefix is case-sensitive; suffix is not
                if image_path_prefix is not None and not locator.startswith(image_path_prefix):
                    continue

                # Although urlparse(p).path preserves the extension on local paths, it will not work for
                # blob file names that contains "#", which will be treated as indication of a query.
                # If the URL is generated via Azure Blob Storage, the "#" char will be properly encoded
                path = urllib.parse.urlparse(locator).path if use_url else locator

                if path.lower().endswith(api_config.IMAGE_SUFFIXES_ACCEPTED):
                    valid_image_paths.append(p)
            image_paths = valid_image_paths
            log.info(('server_job, create_batch_job, length of image_paths provided by user, '
                      f'after filtering to jpg: {len(image_paths)}'))
        
        # apply the first_n and sample_n filters
        if first_n:
            assert first_n > 0, 'parameter first_n is 0.'
            # OK if first_n > total number of images
            image_paths = image_paths[:first_n]

        if sample_n:
            assert sample_n > 0, 'parameter sample_n is 0.'
            if sample_n > len(image_paths):
                msg = ('parameter sample_n specifies more images than '
                       'available (after filtering by other provided params).')
                raise ValueError(msg)

            # sample by shuffling image paths and take the first sample_n images
            log.info('First path before shuffling:', image_paths[0])
            shuffle(image_paths)
            log.info('First path after shuffling:', image_paths[0])
            image_paths = image_paths[:sample_n]

        num_images = len(image_paths)
        log.info(f'server_job, create_batch_job, num_images after applying all filters: {num_images}')
        
        if num_images < 1:
            job_status = get_job_status('completed', (
                'Zero images found in container or in provided list of images '
                'after filtering with the provided parameters.'))
            job_status_table.update_job_status(job_id, job_status)
            return
        if num_images > api_config.MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB:
            job_status = get_job_status(
                'failed',
                (f'The number of images ({num_images}) requested for processing exceeds the maximum '
                 f'accepted {api_config.MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB} in one call'))
            job_status_table.update_job_status(job_id, job_status)
            return

        classifier_params = get_classification_params_from_container(do_classification,hitax_type,do_smoothing)  

        # upload the image list containing their paths (string) to the container
        upload_imgs_to_container(image_paths,job_id)

        job_status = get_job_status('created', f'{num_images} images listed; submitting the job...')
        job_status_table.update_job_status(job_id, job_status)

    except Exception as e:
        job_status = get_job_status('failed', f'Error occurred while preparing the Batch job: {e}')
        job_status_table.update_job_status(job_id, job_status)
        log.error(f'server_job, create_batch_job, Error occurred while preparing the Batch job: {e}')
        return  # do not start monitoring

    try:
        batch_job_manager = BatchJobManager()
        detector_path = (Path(api_config.STORAGE_CONTAINER_MODELS)/api_config.STORAGE_DETECTOR_DIR/api_config.MD_VERSIONS_TO_REL_PATH[model_version]).as_posix()
        
        batch_job_manager.create_job(job_id,
                                     detector_path,
                                     input_container_sas,
                                     use_url,
                                     classifier_params_path=classifier_params.classifier_params_path,
                                     hitax_type=hitax_type,
                                     hitax_output=hitax_output
                                     )

        # submit the tasks to the Batch job
        tasks_start_time = get_utc_time()
        num_tasks, task_ids_failed_to_submit = batch_job_manager.submit_tasks(job_id, 
                                                                              num_images,
                                                                              hitax_type=hitax_type)
        # now request_status moves from created to running
        job_status = get_job_status('running',
                                    (f'Submitted {num_images} images to cluster in {num_tasks} shards. '
                                     f'Number of shards failed to be submitted: {len(task_ids_failed_to_submit)}'))

        # an extra field to allow the monitoring thread to restart after an API restart: total number of tasks
        job_status['num_tasks'] = num_tasks
        # also record the number of images to process for reporting
        job_status['num_images'] = num_images

        job_status_table.update_job_status(job_id, job_status)

    except Exception as e:
        job_status = get_job_status('problem', f'Please contact us. Error occurred while submitting the Batch job: {e}')
        job_status_table.update_job_status(job_id, job_status)
        log.error(f'server_job, create_batch_job, Error occurred while submitting the Batch job: {e}')
        return

    # start the monitor thread with the same name
    try:
        thread = threading.Thread(
            target=monitor_batch_job,
            name=f'job_{job_id}',
            kwargs={
                'job_id': job_id,
                'num_tasks': num_tasks,
                'model_version': model_version,
                'job_name': job_name,
                'job_submission_timestamp': job_submission_timestamp,
                'label_names': classifier_params.classifier_label_names,
                'parent_names': classifier_params.classifier_parent_names,
                'classifier_weight': classifier_params.classifier_weight,
                'tasks_start_time': tasks_start_time,
                'classifier_conf_threshold': classifier_params.classifier_conf_threshold,
                'hitax_type': hitax_type,
                'hitax_output': hitax_output,
                'classifier_hitax_threshold': classifier_params.classifier_hitax_threshold,
                'smoothing_sequence_timegap': classifier_params.smoothing_sequence_timegap,
                'smoothing_min_dominant': classifier_params.smoothing_min_dominant,
                'smoothing_max_nondominant': classifier_params.smoothing_max_nondominant
            }
        )
        thread.start()
    except Exception as e:
        job_status = get_job_status('problem', f'Error occurred while starting the monitoring thread: {e}')
        job_status_table.update_job_status(job_id, job_status)
        log.error(f'server_job, create_batch_job, Error occurred while starting the monitoring thread: {e}')
        return


def monitor_batch_job(job_id: str,
                      num_tasks: int,
                      model_version: str,
                      job_name: str,
                      job_submission_timestamp: str,
                      label_names: Optional[list] = None,
                      parent_names: Optional[list] = None,
                      classifier_weight: Optional[str] = None,
                      tasks_start_time: str = None,
                      classifier_conf_threshold: float = None,
                      hitax_type: str = None,
                      hitax_output: str = api_config.HITAX_OUTPUT_TYPES[0],
                      classifier_hitax_threshold: float = None,
                      smoothing_sequence_timegap: int = None,
                      smoothing_min_dominant: int = None,
                      smoothing_max_nondominant: int = None
                      ):

    job_status_table = JobStatusTable()
    batch_job_manager = BatchJobManager()

    try:
        num_checks = 0

        while True:
            time.sleep(api_config.MONITOR_PERIOD_MINUTES * 60)
            num_checks += 1

            # both succeeded and failed tasks are marked "completed" on Batch
            num_tasks_succeeded, num_tasks_failed = batch_job_manager.get_num_completed_tasks(job_id)
            job_status = get_job_status('running',
                                        (f'Check number {num_checks}, '
                                         f'{num_tasks_succeeded} out of {num_tasks} shards have completed '
                                         f'successfully, {num_tasks_failed} shards have failed.'))
            job_status_table.update_job_status(job_id, job_status)
            log.info(f'job_id {job_id}. '
                f'Check number {num_checks}, {num_tasks_succeeded} out of {num_tasks} shards completed, '
                f'{num_tasks_failed} shards failed.')

            if (num_tasks_succeeded + num_tasks_failed) >= num_tasks:
                break

            if num_checks > api_config.MAX_MONITOR_CYCLES:
                job_status = get_job_status('problem',
                    (
                        f'Job unfinished after {num_checks} x {api_config.MONITOR_PERIOD_MINUTES} minutes, '
                        f'please contact us to retrieve the results. Number of succeeded shards: {num_tasks_succeeded}')
                    )
                job_status_table.update_job_status(job_id, job_status)
                log.warning(f'server_job, create_batch_job, MAX_MONITOR_CYCLES reached, ending thread')
                break  # still aggregate the Tasks' outputs

    except Exception as e:
        job_status = get_job_status('problem', f'Error occurred while monitoring the Batch job: {e}')
        job_status_table.update_job_status(job_id, job_status)
        log.error(f'server_job, create_batch_job, Error occurred while monitoring the Batch job: {e}')
        return

    try:        
        output_sas_url = aggregate_results(job_id, 
                                           model_version, 
                                           job_name, 
                                           job_submission_timestamp, 
                                           label_names,
                                           parent_names,
                                           classifier_weight,
                                           tasks_start_time,
                                           classifier_conf_threshold,
                                           hitax_type,
                                           hitax_output,
                                           classifier_hitax_threshold,
                                           smoothing_sequence_timegap,
                                           smoothing_min_dominant,
                                           smoothing_max_nondominant
                                           )
        # preserving format from before, but SAS URL to 'failed_images' and 'images' are no longer provided
        # failures should be contained in the output entries, indicated by an 'error' field
        msg = {
            'num_failed_shards': num_tasks_failed,
            'output_file_urls': {
                'detections': output_sas_url
            }
        }

        job_status = get_job_status('completed', msg)
        job_status_table.update_job_status(job_id, job_status)

    except Exception as e:
        job_status = get_job_status('problem',
                        f'Please contact us to retrieve the results. Error occurred while aggregating results: {e}')
        job_status_table.update_job_status(job_id, job_status)
        log.error(f'server_job, create_batch_job, Error occurred while aggregating results: {e}')
        return

def perform_smoothing(results, smoothing_sequence_timegap, smoothing_min_dominant, smoothing_max_nondominant,
                      classification_threshold,detection_confidence_threshold):
    sequence_dic = sequence_assignment(results['images'], time_gap=smoothing_sequence_timegap)
    cso = ClassificationSmoothingOptions()
    cso.min_detections_to_overwrite_secondary=smoothing_min_dominant
    cso.max_detections_nondominant_class=smoothing_max_nondominant
    cso.classification_confidence_threshold= classification_threshold
    cso.detection_confidence_threshold= detection_confidence_threshold
    cso.propagate_classifications_through_taxonomy=False

    return smooth_classification_results_sequence_level(results,sequence_dic,
                                                        output_file=None,
                                                        options=cso)


def aggregate_results(job_id: str,
                      model_version: str,
                      job_name: str,
                      job_submission_timestamp: str,
                      label_names: Optional[list] = None,
                      parent_names: Optional[list] = None,
                      classifier_weight: Optional[str] = None,
                      tasks_start_time: str = None,
                      classifier_conf_threshold: float = None,
                      hitax_type: str = None,
                      hitax_output: str = None,
                      classifier_hitax_threshold: float = None,
                      smoothing_sequence_timegap: int = None,
                      smoothing_min_dominant: int = None,
                      smoothing_max_nondominant: int = None
                      ) -> str:
    log.info(f'server_job, aggregate_results starting, job_id: {job_id}')
    
    container_url = sas_blob_utils.build_azure_storage_uri(account=api_config.STORAGE_ACCOUNT_NAME,
                                                           container=api_config.STORAGE_CONTAINER_API)
    # when people download this, the timestamp will have : replaced by _
    output_file_path = f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/{job_id}_detections_{job_name}_{job_submission_timestamp}.json'

    # build classification label map
    classification_categories = None
    if label_names is not None: # do classification
        if parent_names is not None:
            if hitax_type=='hitax classifier' and hitax_output == 'parent':
                # use parent names only
                classification_categories = {str(i): parent_names[i] for i in range(len(parent_names))}
            else:
                # concatenate parent and label names
                total_names = parent_names + label_names
                classification_categories = {str(i): total_names[i] for i in range(len(total_names))}
        else:
            classification_categories = {str(i): label_names[i] for i in range(len(label_names))}
         
    with ContainerClient.from_container_url(container_url,
                                            credential=api_config.STORAGE_ACCOUNT_KEY) as container_client:
        # check if the result blob has already been written (could be another instance of the API / worker thread)
        # and if so, skip aggregating and uploading the results, and just generate the SAS URL, which
        # could be needed still if the previous request_status was `problem`.
        blob_client = container_client.get_blob_client(output_file_path)
        if blob_client.exists():
            log.warning(f'The output file already exists, likely because another monitoring thread already wrote it.')
        else:
            task_outputs_dir = f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/task_outputs/'
            generator = container_client.list_blobs(name_starts_with=task_outputs_dir)

            blobs = [i for i in generator if i.name.endswith('.json')]

            all_results = []
            for blob_props in tqdm(blobs):
                with container_client.get_blob_client(blob_props) as blob_client:
                    stream = io.BytesIO()
                    blob_client.download_blob().readinto(stream)
                    stream.seek(0)
                    task_results = json.load(stream)
                    all_results.extend(task_results)

            # count each category in detection
            count_categories = {}
            for result in all_results:
                if 'detections' in result and len(result['detections']) > 0:
                    for d in result['detections']:
                        category = d['category']
                        count_categories[category] = count_categories.get(category, 0) + 1
            
            api_output = {
                'info': {
                    'format_version': api_config.OUTPUT_FORMAT_VERSION,
                    'detector': api_config.MD_VERSIONS_TO_REL_PATH[model_version],
                    'detection_start_time': job_submission_timestamp,
                    'detection_completion_time': get_utc_time(),
                    'tasks_duration_seconds': int((datetime.now(timezone.utc) - datetime.fromisoformat(job_submission_timestamp)).total_seconds()),
                    'num_images': len(all_results),
                    'num_animals': count_categories.get("1", 0),
                    'num_persons': count_categories.get("2", 0),
                    'detector_megadata': {
                        'megadetector_version': model_version,
                        'detection_conf_threshold': api_config.DETECTION_CONF_THRESHOLD    
                    }
                }
            }
            api_output['detection_categories'] = api_config.DETECTOR_LABEL_MAP
            do_smoothing = False
            if label_names is not None:
                # do classification
                api_output['info']['classifier'] = classifier_weight
                api_output['info']['classifier_metadata'] = {
                    'classifier_conf_threshold': classifier_conf_threshold,
                    }
                if hitax_type in api_config.HITAX_ALL_TYPES[1:]:
                    api_output['info']['classifier_metadata']['higher_taxonomy_type'] = hitax_type.upper()
                    api_output['info']['classifier_metadata']['higher_taxonomy_fill_threshold'] = classifier_hitax_threshold
                if smoothing_sequence_timegap is not None and smoothing_min_dominant is not None and smoothing_max_nondominant is not None:
                    do_smoothing = True
                    api_output['info']['classifier_metadata']['smoothing_sequence_timegap'] = smoothing_sequence_timegap
                    api_output['info']['classifier_metadata']['smoothing_min_dominant'] = smoothing_min_dominant
                    api_output['info']['classifier_metadata']['smoothing_max_nondominant'] = smoothing_max_nondominant
                
                api_output['classification_categories'] = classification_categories


            
                
            api_output['images'] = all_results
                        
            if do_smoothing:
                log.info('server_job, aggregate_results, smoothing started')
                api_output = perform_smoothing(
                    api_output,
                    smoothing_sequence_timegap=smoothing_sequence_timegap,
                    smoothing_min_dominant=smoothing_min_dominant,
                    smoothing_max_nondominant=smoothing_max_nondominant,
                    classification_threshold=classifier_conf_threshold,
                    detection_confidence_threshold=api_config.DETECTION_CONF_THRESHOLD
                )
                log.info('server_job, aggregate_results, smoothing done')

            # upload the output JSON to the Job folder
            api_output_as_bytes = bytes(json.dumps(api_output, ensure_ascii=False, indent=1), encoding='utf-8')
            _ = container_client.upload_blob(name=output_file_path, data=api_output_as_bytes)

    output_sas = generate_blob_sas(
        account_name=api_config.STORAGE_ACCOUNT_NAME,
        container_name=api_config.STORAGE_CONTAINER_API,
        blob_name=output_file_path,
        account_key=api_config.STORAGE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True, write=False),
        expiry=datetime.now(timezone.utc)  + timedelta(days=api_config.OUTPUT_SAS_EXPIRATION_DAYS)
    )
    output_sas_url = sas_blob_utils.build_azure_storage_uri(
        account=api_config.STORAGE_ACCOUNT_NAME,
        container=api_config.STORAGE_CONTAINER_API,
        blob=output_file_path,
        sas_token=output_sas
    )
    log.info(f'server_job, aggregate_results done, job_id: {job_id}')
    log.info(f'output_sas_url: {output_sas_url}')
    return output_sas_url