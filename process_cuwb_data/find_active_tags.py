import datetime

def identify_time_segments(
        timestamps,
        max_gap_duration=datetime.timedelta(seconds=20),
        min_segment_duration=datetime.timedelta(minutes=2)
):
    time_segments = list()
    if len(timestamps) < 2:
        return time_segments
    timestamps_sorted = sorted(timestamps)
    start = timestamps_sorted[0]
    previous_timestamp = timestamps_sorted[0]
    for timestamp in timestamps_sorted[1:]:
        if timestamp - previous_timestamp <= max_gap_duration:
            previous_timestamp = timestamp
            continue
        end = previous_timestamp
        if end - start >= min_segment_duration:
            time_segments.append({'start': start, 'end': end})
        start = timestamp
        previous_timestamp = timestamp
    return time_segments

    