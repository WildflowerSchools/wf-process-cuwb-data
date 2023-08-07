import honeycomb_io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import dateutil
import pathlib

from process_cuwb_data.utils.log import logger


def find_active_tags(
    position_data,
    accelerometer_data,
    max_gap_duration=datetime.timedelta(seconds=20),
    min_segment_duration=datetime.timedelta(minutes=2),
):
    if len(position_data) == 0 or len(accelerometer_data) == 0:
        logger.warning("Empty data; no active periods")
        return pd.DataFrame(
            [],
            columns=[
                "environment_id",
                "device_id",
                "start",
                "end",
            ],
        )
    active_periods_position = find_active_periods(
        data=position_data,
        max_gap_duration=max_gap_duration,
        min_segment_duration=min_segment_duration,
        timestamp_field_name="timestamp",
    )
    active_periods_accelerometer = find_active_periods(
        data=accelerometer_data,
        max_gap_duration=max_gap_duration,
        min_segment_duration=min_segment_duration,
        timestamp_field_name="timestamp",
    )
    active_tags = intersect_active_periods(
        active_periods_a=active_periods_position,
        active_periods_b=active_periods_accelerometer,
    )
    return active_tags


def find_active_periods(
    data,
    max_gap_duration=datetime.timedelta(seconds=20),
    min_segment_duration=datetime.timedelta(minutes=2),
    timestamp_field_name="timestamp",
):
    if len(data) == 0:
        logger.warning("Empty data; no active periods")
        return pd.DataFrame(
            [],
            columns=[
                "environment_id",
                "device_id",
                "start",
                "end",
            ],
        )
    active_period_dfs = []
    for (environment_id, device_id), tag_data in data.groupby(by=["environment_id", "device_id"]):
        time_segments_list = find_time_segments(
            timestamps=tag_data[timestamp_field_name],
            max_gap_duration=max_gap_duration,
            min_segment_duration=min_segment_duration,
        )
        if len(time_segments_list) > 0:
            time_segments = pd.DataFrame(time_segments_list)
            time_segments["environment_id"] = environment_id
            time_segments["device_id"] = device_id
            active_period_dfs.append(time_segments)
    column_names = ["environment_id", "device_id", "start", "end"]
    if len(active_period_dfs) == 0:
        return pd.DataFrame(columns=column_names)
    active_periods = pd.concat(active_period_dfs).reindex(columns=column_names)
    return active_periods


def intersect_active_periods(
    active_periods_a,
    active_periods_b,
):
    if len(active_periods_a) == 0 or len(active_periods_b) == 0:
        logger.warning("Intersection with empty active periods list is empty")
        return pd.DataFrame(
            [],
            columns=[
                "environment_id",
                "device_id",
                "start",
                "end",
            ],
        )
    active_periods = active_periods_a.set_index(["environment_id", "device_id"]).join(
        active_periods_b.set_index(["environment_id", "device_id"]), how="left", lsuffix="_a", rsuffix="_b"
    )

    active_periods["start"] = np.maximum(
        active_periods["start_a"],
        active_periods["start_b"],
    )

    active_periods["end"] = np.minimum(
        active_periods["end_a"],
        active_periods["end_b"],
    )

    active_periods = (
        active_periods.loc[active_periods["start"] < active_periods["end"]]
        .reset_index()
        .reindex(
            columns=[
                "environment_id",
                "device_id",
                "start",
                "end",
            ]
        )
    )
    return active_periods


def find_time_segments(
    timestamps, max_gap_duration=datetime.timedelta(seconds=20), min_segment_duration=datetime.timedelta(minutes=2)
):
    time_segments = []
    if len(timestamps) < 2:
        return time_segments
    timestamps_sorted = sorted(timestamps)
    start = timestamps_sorted[0]
    previous_timestamp = timestamps_sorted[0]
    for timestamp in timestamps_sorted[1:]:
        if timestamp - previous_timestamp <= max_gap_duration:
            previous_timestamp = timestamp
            if timestamp != timestamps_sorted[-1]:
                continue
        end = previous_timestamp
        if end - start >= min_segment_duration:
            time_segments.append({"start": start, "end": end})
        start = timestamp
        previous_timestamp = timestamp
    return time_segments


def visualize_active_tags(
    active_tags,
    start=None,
    end=None,
    device_ids=None,
    timezone_name="UTC",
    require_all=True,
    honeycomb_chunk_size=100,
    honeycomb_client=None,
    honeycomb_uri=None,
    honeycomb_token_uri=None,
    honeycomb_audience=None,
    honeycomb_client_id=None,
    honeycomb_client_secret=None,
    show_visualization=True,
    save_visualization=False,
    save_path=None,
):
    if len(active_tags) == 0:
        logger.warning("No active periods to visualize")
        return
    if start is None:
        start = active_tags["start"].min()
    if end is None:
        end = active_tags["end"].max()
    if device_ids is None:
        device_ids = active_tags["device_id"].unique().tolist()
    device_info = honeycomb_io.fetch_devices(
        device_ids=device_ids,
        output_format="dataframe",
        chunk_size=honeycomb_chunk_size,
        client=honeycomb_client,
        uri=honeycomb_uri,
        token_uri=honeycomb_token_uri,
        audience=honeycomb_audience,
        client_id=honeycomb_client_id,
        client_secret=honeycomb_client_secret,
    )
    entity_assignments = honeycomb_io.fetch_device_entity_assignments_by_device_id(
        device_ids=device_ids,
        start=start,
        end=end,
        require_unique_assignment=True,
        require_all_devices=require_all,
        output_format="dataframe",
        chunk_size=honeycomb_chunk_size,
        client=honeycomb_client,
        uri=honeycomb_uri,
        token_uri=honeycomb_token_uri,
        audience=honeycomb_audience,
        client_id=honeycomb_client_id,
        client_secret=honeycomb_client_secret,
    )
    device_info = device_info.join(entity_assignments.set_index("device_id"), how="left")
    tray_ids = device_info["tray_id"].dropna().unique().tolist()
    material_assignments = honeycomb_io.fetch_tray_material_assignments_by_tray_id(
        tray_ids=tray_ids,
        start=start,
        end=end,
        require_unique_assignment=True,
        require_all_trays=require_all,
        output_format="dataframe",
        chunk_size=honeycomb_chunk_size,
        client=honeycomb_client,
        uri=honeycomb_uri,
        token_uri=honeycomb_token_uri,
        audience=honeycomb_audience,
        client_id=honeycomb_client_id,
        client_secret=honeycomb_client_secret,
    )
    device_info = device_info.join(material_assignments.set_index("tray_id"), how="left", on="tray_id")
    device_info = device_info.sort_values(
        [
            "entity_type",
            "person_type",
            "person_short_name",
            "material_name",
        ]
    )
    device_info["tag_label"] = device_info.apply(generate_tag_label, axis=1)
    device_info["tag_position"] = range(len(device_info), 0, -1)
    fig, ax = plt.subplots()
    for device_id, active_tags_device in active_tags.groupby("device_id"):
        bar_y_position = device_info.loc[device_id, "tag_position"] - 0.4
        bar_y_length = 0.8
        bar_y = (bar_y_position, bar_y_length)
        bars_x = list()
        for period_index, period in active_tags_device.iterrows():
            bar_x_position = period["start"]
            bar_x_length = period["end"] - period["start"]
            bars_x.append((bar_x_position, bar_x_length))
        ax.broken_barh(bars_x, bar_y, label=device_id)
    ax.set_yticks(device_info["tag_position"], labels=device_info["tag_label"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone_name))
    ax.set_title(f"Active tags ({start.astimezone(dateutil.tz.gettz(timezone_name)).strftime('%b %-d, %Y')})")
    ax.set_xlabel(f"Time ({timezone_name})")
    fig.set_size_inches(10.5, 8)
    if save_visualization:
        if save_path is None:
            raise ValueError("Must specify save path")
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show_visualization:
        plt.show()


def generate_tag_label(row):
    if pd.isna(row["entity_type"]):
        return f"Unknown: {row['device_name']}"
    elif row["entity_type"] == "Person":
        return f"{row['person_type'].title()}: {row['person_short_name']}"
    elif row["entity_type"] == "Tray":
        return f"{row['entity_type']}: {row['material_name']}"
    else:
        raise ValueError("Failed to parse row {row}")
