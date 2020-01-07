from database_connection_honeycomb import DatabaseConnectionHoneycomb
import minimal_honeycomb
import pandas as pd

def fetch_cuwb_data(
    environment_name,
    start_time,
    end_time,
    read_chunk_size=2,
    object_type_honeycomb='DEVICE',
    object_id_field_name_honeycomb='device_type',
    object_ids=['UWBTAG']
):
    dbc = DatabaseConnectionHoneycomb(
        environment_name_honeycomb = environment_name,
        time_series_database = True,
        object_database = True,
        object_type_honeycomb = object_type_honeycomb,
        object_id_field_name_honeycomb = object_id_field_name_honeycomb,
        read_chunk_size=read_chunk_size
    )
    data = dbc.fetch_data_object_time_series(
        start_time = start_time,
        end_time = end_time,
        object_ids = object_ids
    )
    df = pd.DataFrame(data)
    df = df.loc[df['type'] == 'position'].copy()
    df = df.reindex(columns=[
        'serial_number',
        'timestamp_secondary',
        'x',
        'y',
        'z',
        'anchor_count',
        'quality'
    ])
    df.rename(
        columns = {
            'timestamp_secondary': 'timestamp'
        },
        inplace=True
    )
    df['x_meters'] = df['x']/1000.0
    df['y_meters'] = df['y']/1000.0
    df['z_meters'] = df['z']/1000.0
    df.drop(
        columns=['x', 'y', 'z'],
        inplace=True
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df
