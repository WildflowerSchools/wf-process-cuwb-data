import honeycomb_rds_client
import honeycomb_io
import pandas as pd
import json
import hashlib
import datetime
import pathlib

from process_cuwb_data.utils.log import logger


def fetch_position_data(
    start,
    end,
    device_ids=None,
    part_numbers=None,
    serial_numbers=None,
    tag_ids=None,
    names=None,
    environment_id=None,
    environment_name=None,
    include_device_info=False,
    include_entity_info=False,
    include_material_info=False,
    use_cache=True,
    cache_directory="/data/uwb_data",
    rds_connection=None,
    rds_dbname=None,
    rds_user=None,
    rds_password=None,
    rds_host=None,
    rds_port=None,
    honeycomb_chunk_size=100,
    honeycomb_client=None,
    honeycomb_uri=None,
    honeycomb_token_uri=None,
    honeycomb_audience=None,
    honeycomb_client_id=None,
    honeycomb_client_secret=None,
):
    if use_cache:
        file_path = generate_file_path(
            filename_prefix="position_data",
            start=start,
            end=end,
            device_ids=device_ids,
            part_numbers=part_numbers,
            serial_numbers=serial_numbers,
            tag_ids=tag_ids,
            names=names,
            environment_id=environment_id,
            environment_name=environment_name,
            include_device_info=include_device_info,
            include_entity_info=include_entity_info,
            include_material_info=include_material_info,
            cache_directory=cache_directory,
            honeycomb_chunk_size=honeycomb_chunk_size,
            honeycomb_client=honeycomb_client,
            honeycomb_uri=honeycomb_uri,
            honeycomb_token_uri=honeycomb_token_uri,
            honeycomb_audience=honeycomb_audience,
            honeycomb_client_id=honeycomb_client_id,
            honeycomb_client_secret=honeycomb_client_secret,
        )
        if file_path.is_file():
            position_data = pd.read_pickle(file_path)
            logger.info(f"File {file_path} exists locally. Fetching from local")
            return position_data
    logger.info("Fetching data from RDS database")
    client = honeycomb_rds_client.HoneycombRDSClient(
        dbname=rds_dbname, user=rds_user, password=rds_password, host=rds_host, port=rds_port
    )
    position_data = client.fetch_position_data(
        start=start,
        end=end,
        device_ids=device_ids,
        part_numbers=part_numbers,
        serial_numbers=serial_numbers,
        tag_ids=tag_ids,
        names=names,
        environment_id=environment_id,
        environment_name=environment_name,
        include_device_info=include_device_info,
        include_entity_info=include_entity_info,
        include_material_info=include_material_info,
        connection=rds_connection,
        honeycomb_chunk_size=honeycomb_chunk_size,
        honeycomb_client=honeycomb_client,
        honeycomb_uri=honeycomb_uri,
        honeycomb_token_uri=honeycomb_token_uri,
        honeycomb_audience=honeycomb_audience,
        honeycomb_client_id=honeycomb_client_id,
        honeycomb_client_secret=honeycomb_client_secret,
    )
    if use_cache:
        logger.info(f"Saving data locally as {file_path}")
        position_data.to_pickle(file_path)
    return position_data


def fetch_accelerometer_data(
    start,
    end,
    device_ids=None,
    part_numbers=None,
    serial_numbers=None,
    tag_ids=None,
    names=None,
    environment_id=None,
    environment_name=None,
    include_device_info=False,
    include_entity_info=False,
    include_material_info=False,
    use_cache=True,
    cache_directory="/data/uwb_data",
    rds_connection=None,
    rds_dbname=None,
    rds_user=None,
    rds_password=None,
    rds_host=None,
    rds_port=None,
    honeycomb_chunk_size=100,
    honeycomb_client=None,
    honeycomb_uri=None,
    honeycomb_token_uri=None,
    honeycomb_audience=None,
    honeycomb_client_id=None,
    honeycomb_client_secret=None,
):
    if use_cache:
        file_path = generate_file_path(
            filename_prefix="accelerometer_data",
            start=start,
            end=end,
            device_ids=device_ids,
            part_numbers=part_numbers,
            serial_numbers=serial_numbers,
            tag_ids=tag_ids,
            names=names,
            environment_id=environment_id,
            environment_name=environment_name,
            include_device_info=include_device_info,
            include_entity_info=include_entity_info,
            include_material_info=include_material_info,
            cache_directory=cache_directory,
            honeycomb_chunk_size=honeycomb_chunk_size,
            honeycomb_client=honeycomb_client,
            honeycomb_uri=honeycomb_uri,
            honeycomb_token_uri=honeycomb_token_uri,
            honeycomb_audience=honeycomb_audience,
            honeycomb_client_id=honeycomb_client_id,
            honeycomb_client_secret=honeycomb_client_secret,
        )
        if file_path.is_file():
            accelerometer_data = pd.read_pickle(file_path)
            logger.info(f"File {file_path} exists locally. Fetching from local")
            return accelerometer_data
    logger.info("Fetching data from RDS database")
    client = honeycomb_rds_client.HoneycombRDSClient(
        dbname=rds_dbname, user=rds_user, password=rds_password, host=rds_host, port=rds_port
    )
    accelerometer_data = client.fetch_accelerometer_data(
        start=start,
        end=end,
        device_ids=device_ids,
        part_numbers=part_numbers,
        serial_numbers=serial_numbers,
        tag_ids=tag_ids,
        names=names,
        environment_id=environment_id,
        environment_name=environment_name,
        include_device_info=include_device_info,
        include_entity_info=include_entity_info,
        include_material_info=include_material_info,
        connection=rds_connection,
        honeycomb_chunk_size=honeycomb_chunk_size,
        honeycomb_client=honeycomb_client,
        honeycomb_uri=honeycomb_uri,
        honeycomb_token_uri=honeycomb_token_uri,
        honeycomb_audience=honeycomb_audience,
        honeycomb_client_id=honeycomb_client_id,
        honeycomb_client_secret=honeycomb_client_secret,
    )
    if use_cache:
        logger.info(f"Saving data locally as {file_path}")
        accelerometer_data.to_pickle(file_path)
    return accelerometer_data


def generate_file_path(
    filename_prefix,
    start,
    end,
    device_ids=None,
    part_numbers=None,
    serial_numbers=None,
    tag_ids=None,
    names=None,
    environment_id=None,
    environment_name=None,
    include_device_info=False,
    include_entity_info=False,
    include_material_info=False,
    cache_directory="/data/uwb_data",
    honeycomb_chunk_size=100,
    honeycomb_client=None,
    honeycomb_uri=None,
    honeycomb_token_uri=None,
    honeycomb_audience=None,
    honeycomb_client_id=None,
    honeycomb_client_secret=None,
):
    if environment_id is None:
        if environment_name is None:
            raise ValueError("Must specify either environment ID or environment name")
        environment_id = honeycomb_io.fetch_environment_id(environment_name=environment_name)
    start_string = start.astimezone(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    end_string = end.astimezone(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    device_ids = honeycomb_io.fetch_device_ids(
        device_types=["UWBTAG"],
        device_ids=device_ids,
        part_numbers=part_numbers,
        serial_numbers=serial_numbers,
        tag_ids=tag_ids,
        names=names,
        environment_id=environment_id,
        environment_name=None,
        start=start,
        end=end,
        chunk_size=honeycomb_chunk_size,
        client=honeycomb_client,
        uri=honeycomb_uri,
        token_uri=honeycomb_token_uri,
        audience=honeycomb_audience,
        client_id=honeycomb_client_id,
        client_secret=honeycomb_client_secret,
    )
    arguments_hash = generate_arguments_hash(
        start=start,
        end=end,
        environment_id=environment_id,
        device_ids=device_ids,
        include_device_info=include_device_info,
        include_entity_info=include_entity_info,
        include_material_info=include_material_info,
    )
    file_path = pathlib.Path(cache_directory) / ".".join(
        ["_".join([filename_prefix, environment_id, start_string, end_string, arguments_hash]), "pkl"]
    )
    return file_path


def generate_arguments_hash(
    start,
    end,
    environment_id,
    device_ids,
    include_device_info,
    include_entity_info,
    include_material_info,
):
    arguments_normalized = (
        start.timestamp(),
        end.timestamp(),
        environment_id,
        tuple(sorted(device_ids)),
        include_device_info,
        include_entity_info,
        include_material_info,
    )
    arguments_serialized = json.dumps(arguments_normalized)
    arguments_hash = hashlib.sha256(arguments_serialized.encode()).hexdigest()
    return arguments_hash
