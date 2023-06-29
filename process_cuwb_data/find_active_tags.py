import datetime

import pandas as pd


def find_active_tags(
    accelerometer_data,
    max_gap_duration=datetime.timedelta(seconds=20),
    min_segment_duration=datetime.timedelta(minutes=2),
):
    active_tag_dfs = []
    for device_id, accelerometer_data_tag in accelerometer_data.groupby("device_id"):
        time_segments_tag_list = find_time_segments(
            timestamps=accelerometer_data_tag["timestamp"],
            max_gap_duration=max_gap_duration,
            min_segment_duration=min_segment_duration,
        )
        if len(time_segments_tag_list) > 0:
            time_segments = pd.DataFrame(time_segments_tag_list)
            time_segments["device_id"] = device_id
            active_tag_dfs.append(time_segments)
    column_names = ["device_id", "start", "end"]
    if len(active_tag_dfs) == 0:
        return pd.DataFrame(columns=column_names)
    active_tags = pd.concat(active_tag_dfs).reindex(columns=column_names)
    return active_tags


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
