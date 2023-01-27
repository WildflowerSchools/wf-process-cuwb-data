from functools import lru_cache
import os

import honeycomb_io
import minimal_honeycomb


class HoneycombCachingClient:
    __instance = None

    def __new__(cls):
        print(cls.__instance)
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(
        self,
        url=os.getenv("HONEYCOMB_URI", "https://honeycomb.api.wildflower-tech.org/graphql"),
        auth_domain=os.getenv("HONEYCOMB_DOMAIN", os.getenv("AUTH0_DOMAIN", "wildflowerschools.auth0.com")),
        auth_client_id=os.getenv("HONEYCOMB_CLIENT_ID", os.getenv("AUTH0_CLIENT_ID", None)),
        auth_client_secret=os.getenv("HONEYCOMB_CLIENT_SECRET", os.getenv("AUTH0_CLIENT_SECRET", None)),
        auth_audience=os.getenv("HONEYCOMB_AUDIENCE", os.getenv("API_AUDIENCE", "wildflower-tech.org")),
    ):
        if auth_client_id is None:
            raise ValueError("HONEYCOMB_CLIENT_ID (or AUTH0_CLIENT_ID) is required")
        if auth_client_secret is None:
            raise ValueError("HONEYCOMB_CLIENT_SECRET (or AUTH0_CLIENT_SECRET) is required")

        token_uri = os.getenv("HONEYCOMB_TOKEN_URI", f"https://{auth_domain}/oauth/token")

        self.client: minimal_honeycomb.MinimalHoneycombClient = honeycomb_io.generate_client(
            uri=url,
            token_uri=token_uri,
            audience=auth_audience,
            client_id=auth_client_id,
            client_secret=auth_client_secret,
        )

        self.client_params = {
            "client": self.client,
            "uri": url,
            "token_uri": token_uri,
            "audience": auth_audience,
            "client_id": auth_client_id,
            "client_secret": auth_client_secret,
        }

    @lru_cache(maxsize=10)
    def fetch_camera_devices(self, environment_id=None, environment_name=None, start=None, end=None, chunk_size=200):
        return honeycomb_io.fetch_devices(
            device_types=honeycomb_io.DEFAULT_CAMERA_DEVICE_TYPES,
            environment_id=environment_id,
            environment_name=environment_name,
            start=start,
            end=end,
            output_format="dataframe",
            chunk_size=chunk_size,
            **self.client_params,
        )

    @lru_cache(maxsize=50)
    def fetch_camera_calibrations(self, camera_ids: tuple, start=None, end=None, chunk_size=100):
        return honeycomb_io.fetch_camera_calibrations(
            camera_ids=list(camera_ids), start=start, end=end, chunk_size=chunk_size, **self.client_params
        )

    @lru_cache(maxsize=50)
    def fetch_camera_info(self, environment_name, start=None, end=None, chunk_size=100):
        return honeycomb_io.fetch_camera_info(
            environment_name=environment_name, start=start, end=end, chunk_size=chunk_size
        )

    @lru_cache(maxsize=20)
    def fetch_environment_by_name(self, environment_name):
        return honeycomb_io.fetch_environment_by_name(environment_name)

    @lru_cache(maxsize=50)
    def fetch_environment_id(self, environment_name):
        return honeycomb_io.fetch_environment_id(environment_name=environment_name)

    @lru_cache()
    def fetch_all_environments(self):
        return honeycomb_io.fetch_all_environments(output_format="dataframe", **self.client_params)
