import datetime
from typing import List, Optional

from .honeycomb_service import HoneycombCachingClient


class CameraHelper:
    def __init__(
        self,
        environment_id: str,
        environment_name: str,
        start: datetime.datetime,
        end: datetime.datetime,
        chunk_size: int = 100,
        client=None,
        uri=None,
        token_uri=None,
        audience=None,
        client_id=None,
        client_secret=None,
    ):
        if environment_id is None and environment_name is None:
            raise ValueError("Must specify either environment_name ID or environment_name name")

        if start is None or end is None:
            raise ValueError("Must specify both start and end timestamps")

        honeycomb_caching_client = HoneycombCachingClient()

        self.environment_id = environment_id
        self.environment_name = environment_name

        self.client_params = {
            "chunk_size": chunk_size,
            "client": client,
            "uri": uri,
            "token_uri": token_uri,
            "audience": audience,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        if self.environment_id is None:
            e = honeycomb_caching_client.fetch_environment_by_name(environment_name=environment_name)
            if e is None:
                raise ValueError(f"Couldn't find environment_name: {environment_name}")
            self.environment_id = e["environment_id"]

        if self.environment_name is None:
            df_e = honeycomb_caching_client.fetch_all_environments()
            if df_e is None:
                raise ValueError(f"Couldn't find environment_name by id: {self.environment_id}")
            self.environment_name = df_e.loc[e["environment_id"] == self.environment_id][0]

        self.start = start
        self.end = end
        self.df_camera_info = honeycomb_caching_client.fetch_camera_info(
            environment_name=self.environment_name, start=start, end=end
        )
        self.camera_calibrations = None

    def get_camera_calibrations(self, camera_device_ids: Optional[List[str]] = None):
        honeycomb_caching_client = HoneycombCachingClient()

        _camera_device_ids = camera_device_ids
        if camera_device_ids is None:
            _camera_device_ids = self.get_camera_ids()

        self.camera_calibrations = honeycomb_caching_client.fetch_camera_calibrations(
            camera_ids=tuple(_camera_device_ids), start=self.start, end=self.end
        )

        return self.camera_calibrations

    def get_camera_ids(self):
        return self.df_camera_info.index.to_list()

    def get_camera_id_by_name(self, camera_name):
        m = self.df_camera_info[self.df_camera_info["device_name"] == camera_name]
        if len(m) == 0:
            return None
        if len(m) > 1:
            raise ValueError(f"More than one camera with camera name {camera_name} found")

        return m.index[0]

    def get_camera_name_by_id(self, camera_id):
        m = self.df_camera_info[camera_id]
        if len(m) == 0:
            return None

        return m.index[0]

    def get_camera_info_dict(self):
        return self.df_camera_info.to_dict(orient="index")
