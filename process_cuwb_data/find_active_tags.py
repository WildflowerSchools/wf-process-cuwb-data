import datetime

import pandas as pd


def find_active_periods(
    data,
    max_gap_duration=datetime.timedelta(seconds=20),
    min_segment_duration=datetime.timedelta(minutes=2),
    timestamp_field_name='timestamp',
):
    active_period_dfs = []
    for device_id, tag_data in data.groupby("device_id"):
        time_segments_list = find_time_segments(
            timestamps=tag_data[timestamp_field_name],
            max_gap_duration=max_gap_duration,
            min_segment_duration=min_segment_duration,
        )
        if len(time_segments_list) > 0:
            time_segments = pd.DataFrame(time_segments_list)
            time_segments["device_id"] = device_id
            active_period_dfs.append(time_segments)
    column_names = ["device_id", "start", "end"]
    if len(active_period_dfs) == 0:
        return pd.DataFrame(columns=column_names)
    active_periods = pd.concat(active_period_dfs).reindex(columns=column_names)
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
