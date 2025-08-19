# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A class wrapping the Azure App Configuration client to get configurations
for each instance of the API.
"""
import logging
import os

from server_api_config import SENTINEL

# from azure.appconfiguration import AzureAppConfigurationClient


log = logging.getLogger(os.environ['FLASK_APP'])


class AppConfig:
    """Wrapper around the Azure App Configuration client"""

    def __init__(self):
        # self.client = AzureAppConfigurationClient.from_connection_string(APP_CONFIG_CONNECTION_STR) # Azure App Configuration instance
        # self.api_instance = API_INSTANCE_NAME # cp

        # sentinel should change if new configurations are available
        self.sentinel = self._get_sentinel() # 'awc'
        self.allowlist = self._get_allowlist() # ['awc']

    def _get_sentinel(self):
        # return self.client.get_configuration_setting(key='batch_api:sentinel').value
        return SENTINEL

    def _get_allowlist(self):
        # filtered_listed = self.client.list_configuration_settings(key_filter='batch_api_allow:*')
        # allowlist = []
        # for item in filtered_listed:
        #     if item.value == self.api_instance:
        #         allowlist.append(item.key.split('batch_api_allow:')[1])
        # return allowlist 
        return [SENTINEL]

    def get_allowlist(self):
        # try:
        #     cur_sentinel = self._get_sentinel()
        #     if cur_sentinel == self.sentinel:
        #         # configs have not changed
        #         return self.allowlist
        #     else:
        #         self.sentinel = cur_sentinel
        #         self.allowlist = self._get_allowlist()
        #         return self.allowlist

        # except Exception as e:
        #     log.error(f'AppConfig, get_allowlist, exception so using old allowlist: {e}')
        #     return self.allowlist
        return self.allowlist
