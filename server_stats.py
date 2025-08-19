from azure.storage.blob import ContainerClient
import server_api_config as api_config
import sas_blob_utils
from tqdm import tqdm
import json
container_url = sas_blob_utils.build_azure_storage_uri(account=api_config.STORAGE_ACCOUNT_NAME,
                                                           container=api_config.STORAGE_CONTAINER_API)

with ContainerClient.from_container_url(container_url,
                                            credential=api_config.STORAGE_ACCOUNT_KEY) as container_client:

    task_outputs_dir = f'api_{api_config.API_INSTANCE_NAME}'
    generator = container_client.list_blobs(name_starts_with=task_outputs_dir)

    blobs = [i for i in generator if i.name.endswith('_images.json') or '_detections_' in i.name]
    blobs = sorted(blobs, key=lambda x: x.name)

    print(len(blobs), 'blobs found in the container')
    print([i.name for i in blobs])

    # all_results = []
    # for blob_props in tqdm(blobs):
    #     with container_client.get_blob_client(blob_props) as blob_client:
    #         stream = io.BytesIO()
    #         blob_client.download_blob().readinto(stream)
    #         stream.seek(0)
    #         task_results = json.load(stream)
    #         all_results.extend(task_results)


    # api_output = {
    #     'info': {
    #         'format_version': api_config.OUTPUT_FORMAT_VERSION,
    #         'detector': api_config.MD_VERSIONS_TO_REL_PATH[model_version],
    #         'detection_start_time': job_submission_timestamp,
    #         'detection_completion_time': get_utc_time(),
    #         'tasks_duration_seconds': int((datetime.now(timezone.utc) - datetime.fromisoformat(job_submission_timestamp)).total_seconds()),
    #         'detector_megadata': {
    #             'megadetector_version': model_version,
    #             'detection_conf_threshold': api_config.DETECTION_CONF_THRESHOLD    
    #         }
    #     }
    # }
    # api_output['detection_categories']= api_config.DETECTOR_LABEL_MAP

    # if label_names:
    #     api_output['info']['classifier'] = classifier_weight
    #     api_output['info']['classifier_metadata'] = {
    #         'classification_conf_threshold': api_config.CLASSIFIER_CONF_THRESHOLD
    #         }
    #     api_output['classification_categories'] = classification_categories
    
    # api_output['images']=all_results

    # # upload the output JSON to the Job folder
    # api_output_as_bytes = bytes(json.dumps(api_output, ensure_ascii=False, indent=1), encoding='utf-8')
    # _ = container_client.upload_blob(name=output_file_path, data=api_output_as_bytes)