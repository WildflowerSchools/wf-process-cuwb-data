import datetime
import uuid
import functools

import cv_utils
import honeycomb_io
import numpy as np
import pandas as pd
import pytz

from .camera_helper import CameraHelper
from .camera_uwb_line_of_sight import CameraUWBLineOfSight
from .honeycomb_service import HoneycombCachingClient
from .utils.log import logger


def parse_tray_events(
    df_tray_events,
    environment_id=None,
    environment_name=None,
    time_zone=None,
    camera_device_ids=None,
    default_camera_device_id=None,
    default_camera_name=None,
    camera_calibrations=None,
    position_window_seconds=4,
    imputed_z_position=1.0,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
):
    if df_tray_events is None:
        return None

    if environment_id is None and environment_name is None:
        raise ValueError("Environment info must be provided to parse tray events")

    if time_zone is None:
        raise ValueError("Timezone is required")

    if time_zone not in pytz.all_timezones_set:
        raise ValueError(f"Timezone '{time_zone}' is invalid")

    client_params = {
        "chunk_size": chunk_size,
        "client": client,
        "uri": uri,
        "token_uri": token_uri,
        "audience": audience,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    camera_helper = CameraHelper(
        environment_id=environment_id,
        environment_name=environment_name,
        start=df_tray_events["start"].min(),
        end=df_tray_events["end"].max(),
        **client_params,
    )

    if camera_calibrations is None:
        camera_calibrations = camera_helper.get_camera_calibrations(camera_device_ids=camera_device_ids)

    if default_camera_device_id is None:
        if default_camera_name is None or default_camera_name == "":
            default_camera_name = camera_helper.df_camera_info.iloc[1]["device_name"]
            # raise ValueError("Must specify default camera device ID or name")

        default_camera_device_id = camera_helper.get_camera_id_by_name(default_camera_name)
        if default_camera_device_id is None:
            raise ValueError(f"Default camera name {default_camera_name} not found")

    person_ids = df_tray_events["person_id"].dropna().unique().tolist()
    person_info = honeycomb_io.fetch_persons(person_ids=person_ids, output_format="dataframe", **client_params)
    person_info = person_info.rename(
        columns={
            column_name: (("person_" + column_name) if not column_name.startswith("person_") else column_name)
            for column_name in person_info.columns
        }
    ).astype("object")
    person_info = person_info.where(pd.notnull(person_info), None)

    df_tray_events = df_tray_events.copy()
    df_tray_events["id"] = [str(uuid.uuid4()) for _ in range(len(df_tray_events))]
    df_tray_events["date"] = df_tray_events["start"].dt.tz_convert(time_zone).apply(lambda x: x.date())
    df_tray_events["timestamp"] = df_tray_events["start"]
    df_tray_events = df_tray_events.drop(columns=person_info.columns, errors="ignore").join(
        person_info, how="left", on="person_id"
    )
    df_tray_events = determine_best_cameras_for_trays(
        df=df_tray_events,
        time_fields=["start", "end"],
        camera_device_dict=camera_helper.get_camera_info_dict(),
        environment_id=environment_id,
        environment_name=environment_name,
        camera_device_ids=camera_device_ids,
        default_camera_device_id=default_camera_device_id,
        camera_calibrations=camera_calibrations,
        position_window_seconds=position_window_seconds,
        imputed_z_position=imputed_z_position,
        **client_params,
    )

    df_tray_events["duration_seconds"] = (df_tray_events["end"] - df_tray_events["start"]).dt.total_seconds()
    df_tray_events["description"] = df_tray_events.apply(
        lambda event: describe_tray_event(
            timestamp=event["timestamp"],
            material_name=event["material_name"],
            person_name=event["person_name"],
            interaction_type=event["interaction_type"],
            time_zone=time_zone,
        ),
        axis=1,
    )
    df_tray_events["anonymized_description"] = df_tray_events.apply(
        lambda event: describe_tray_event(
            timestamp=event["timestamp"],
            material_name=event["material_name"],
            person_name=event["person_anonymized_name"],
            interaction_type=event["interaction_type"],
            time_zone=time_zone,
        ),
        axis=1,
    )
    df_tray_events = df_tray_events.reindex(
        columns=[
            "id",
            "date",
            "timestamp",
            "interaction_type",
            "tray_device_id",
            "material_id",
            "material_name",
            "duration_seconds",
            "person_device_id",
            "person_id",
            "person_type",
            "person_name",
            "person_first_name",
            "person_last_name",
            "person_nickname",
            "person_short_name",
            "person_anonymized_name",
            "person_anonymized_first_name",
            "person_anonymized_last_name",
            "person_anonymized_nickname",
            "person_anonymized_short_name",
            "person_transparent_classroom_id",
            "start",
            "end",
            "description",
            "anonymized_description",
            "best_camera_device_id_start",
            "best_camera_name_start",
            "all_in_middle_camera_device_ids_start",
            "all_in_middle_camera_names_start",
            "all_in_frame_camera_device_ids_start",
            "all_in_frame_camera_names_start",
            "best_camera_device_id_end",
            "best_camera_name_end",
            "all_in_middle_camera_device_ids_end",
            "all_in_middle_camera_names_end",
            "all_in_frame_camera_device_ids_end",
            "all_in_frame_camera_names_end",
        ]
    )
    df_tray_events.sort_values("timestamp", inplace=True)
    return df_tray_events


def determine_best_cameras_for_trays(
    df,
    camera_device_dict,
    time_fields=None,
    environment_id=None,
    environment_name=None,
    camera_device_ids=None,
    default_camera_device_id=None,
    camera_calibrations=None,
    position_window_seconds=4,
    imputed_z_position=1.0,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
):
    if time_fields is None:
        time_fields = []

    if df is None:
        return None

    if not pd.Series(time_fields + ["tray_device_id"]).isin(df.columns).all():
        raise ValueError(
            f"Unable to determine best cameras, the supplied dataframe is missing some or all of: '{time_fields + ['tray_device_id']}'"
        )

    if environment_id is None and environment_name is None:
        raise ValueError("Must specify either environment_name ID or environment_name name")

    best_camera_partial = functools.partial(
        CameraUWBLineOfSight,
        default_camera_device_id=default_camera_device_id,
        environment_id=environment_id,
        environment_name=environment_name,
        camera_device_ids=camera_device_ids,
        camera_calibrations=camera_calibrations,
        position_window_seconds=position_window_seconds,
        imputed_z_position=imputed_z_position,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
    )

    def generate_camera_recommendations(event, event_time_field):
        event[f"best_camera_device_id_{event_time_field}"] = None
        event[f"best_camera_name_{event_time_field}"] = None
        event[f"all_in_middle_camera_device_ids_{event_time_field}"] = None
        event[f"all_in_middle_camera_names_{event_time_field}"] = None
        event[f"all_in_frame_camera_device_ids_{event_time_field}"] = None
        event[f"all_in_frame_camera_names_{event_time_field}"] = None

        if pd.isnull(event[event_time_field]):
            return event

        try:
            uwb_line_of_sight = best_camera_partial(
                timestamp=event[event_time_field],
                tag_device_id=event["tray_device_id"],
            )
        except ValueError:
            return event

        event[f"best_camera_device_id_{event_time_field}"] = uwb_line_of_sight.best_camera_view_device_id()
        event[f"best_camera_name_{event_time_field}"] = camera_device_dict.get(
            event[f"best_camera_device_id_{event_time_field}"]
        )["device_name"]
        event[
            f"all_in_middle_camera_device_ids_{event_time_field}"
        ] = uwb_line_of_sight.all_in_middle_camera_views_device_ids()
        event[f"all_in_middle_camera_names_{event_time_field}"] = list(
            map(
                lambda camera_device_id: camera_device_dict.get(camera_device_id)["device_name"],
                event[f"all_in_middle_camera_device_ids_{event_time_field}"],
            )
        )
        event[
            f"all_in_frame_camera_device_ids_{event_time_field}"
        ] = uwb_line_of_sight.all_in_frame_camera_views_device_ids()
        event[f"all_in_frame_camera_names_{event_time_field}"] = list(
            map(
                lambda camera_device_id: camera_device_dict.get(camera_device_id)["device_name"],
                event[f"all_in_frame_camera_device_ids_{event_time_field}"],
            )
        )

        return event

    for time_field in time_fields:
        df = df.apply(generate_camera_recommendations, event_time_field=time_field, axis=1)

    return df


def describe_tray_event(timestamp, material_name, person_name, interaction_type, time_zone):
    time_string = timestamp.tz_convert(time_zone).strftime("%I:%M %p")
    person_string = person_name if pd.notnull(person_name) else "An unknown person"
    if interaction_type == "CARRYING_FROM_SHELF":
        description_text = f"{person_string} took the {material_name} tray from shelf"
    elif interaction_type == "CARRYING_TO_SHELF":
        description_text = f"{person_string} put the {material_name} tray back on the shelf"
    elif interaction_type == "CARRYING_BETWEEN_NON_SHELF_LOCATIONS":
        description_text = f"{person_string} moved the {material_name} tray"
    elif interaction_type == "CARRYING_FROM_AND_TO_SHELF":
        description_text = f"{person_string} took the {material_name} tray from the shelf and immediately put it back"
    else:
        raise ValueError(f"Unexpected interaction type: '{interaction_type}'")
    description = f"{time_string}: {description_text}"
    return description


def generate_material_events(
    df_parsed_tray_events,
    environment_id=None,
    environment_name=None,
    time_zone=None,
    camera_device_ids=None,
    default_camera_device_id=None,
    default_camera_name=None,
    camera_calibrations=None,
    position_window_seconds=4,
    imputed_z_position=1.0,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
):
    if df_parsed_tray_events is None:
        return None

    if environment_id is None and environment_name is None:
        raise ValueError("Environment info must be provided to generate material events")

    if time_zone is None:
        raise ValueError("Timezone is required")

    if time_zone not in pytz.all_timezones_set:
        raise ValueError(f"Timezone '{time_zone}' is invalid")

    df_parsed_tray_events = df_parsed_tray_events.copy()
    material_events_list = []
    for _, parsed_tray_events_date_tray in df_parsed_tray_events.groupby(["date", "tray_device_id"]):
        material_events_list.extend(generate_material_events_date_tray(parsed_tray_events_date_tray))

    if len(material_events_list) == 0:
        return None

    client_params = {
        "chunk_size": chunk_size,
        "client": client,
        "uri": uri,
        "token_uri": token_uri,
        "audience": audience,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    material_events = pd.DataFrame(material_events_list)

    camera_helper = CameraHelper(
        environment_id=environment_id,
        environment_name=environment_name,
        start=material_events["start"].min(),
        end=material_events["end"].max(),
        **client_params,
    )

    if camera_calibrations is None:
        camera_calibrations = camera_helper.get_camera_calibrations(camera_device_ids=camera_device_ids)

    if default_camera_device_id is None:
        if default_camera_name is None:
            logger.warning("Default camera device ID or name not provided")

        default_camera_device_id = camera_helper.get_camera_id_by_name(default_camera_name)
        if default_camera_device_id is None:
            logger.warning(f"Default camera name {default_camera_name} not found")

    material_events["id"] = [str(uuid.uuid4()) for _ in range(len(material_events))]
    material_events["timestamp"] = material_events.apply(
        lambda row: row["start"] if pd.notnull(row["start"]) else row["end"], axis=1
    )
    material_events["middle_of_event"] = material_events.apply(
        lambda row: (row["end"] - (row["end"] - row["start"]) / 2)
        if (pd.notnull(row["start"]) and pd.notnull(row["end"]))
        else np.nan,
        axis=1,
    )
    material_events["duration_seconds"] = material_events.apply(
        lambda row: (row["end"] - row["start"]).total_seconds()
        if (pd.notnull(row["start"]) and pd.notnull(row["end"]))
        else 0,
        axis=1,
    )
    material_events = determine_best_cameras_for_trays(
        df=material_events,
        time_fields=["middle_of_event"],
        camera_device_dict=camera_helper.get_camera_info_dict(),
        environment_id=environment_id,
        environment_name=environment_name,
        camera_device_ids=camera_device_ids,
        default_camera_device_id=default_camera_device_id,
        camera_calibrations=camera_calibrations,
        position_window_seconds=position_window_seconds,
        imputed_z_position=imputed_z_position,
        **client_params,
    )

    material_events["description"] = material_events.apply(
        lambda event: describe_material_event(
            timestamp=event["timestamp"],
            material_name=event["material_name"],
            start=event["start"],
            person_name_from_shelf=event["person_name_from_shelf"],
            end=event["end"],
            person_name_to_shelf=event["person_name_to_shelf"],
            duration_seconds=event["duration_seconds"],
            time_zone=time_zone,
        ),
        axis=1,
    )
    material_events["anonymized_description"] = material_events.apply(
        lambda event: describe_material_event(
            timestamp=event["timestamp"],
            material_name=event["material_name"],
            start=event["start"],
            person_name_from_shelf=event["person_anonymized_name_from_shelf"],
            end=event["end"],
            person_name_to_shelf=event["person_anonymized_name_to_shelf"],
            duration_seconds=event["duration_seconds"],
            time_zone=time_zone,
        ),
        axis=1,
    )
    material_events = material_events.reindex(
        columns=[
            "id",
            "date",
            "timestamp",
            "tray_device_id",
            "material_id",
            "material_name",
            "duration_seconds",
            "start",
            "id_from_shelf",
            "person_device_id_from_shelf",
            "person_id_from_shelf",
            "person_name_from_shelf",
            "person_anonymized_name_from_shelf",
            "person_type_from_shelf",
            "best_camera_device_id_from_shelf",
            "best_camera_name_from_shelf",
            "all_in_middle_camera_device_ids_from_shelf",
            "all_in_middle_camera_names_from_shelf",
            "all_in_frame_camera_device_ids_from_shelf",
            "all_in_frame_camera_names_from_shelf",
            "best_camera_device_id_middle_of_event",
            "best_camera_name_middle_of_event",
            "all_in_middle_camera_device_ids_middle_of_event",
            "all_in_middle_camera_names_middle_of_event",
            "all_in_frame_camera_device_ids_middle_of_event",
            "all_in_frame_camera_names_middle_of_event",
            "id_to_shelf",
            "person_device_id_to_shelf",
            "person_id_to_shelf",
            "person_name_to_shelf",
            "person_anonymized_name_to_shelf",
            "person_type_to_shelf",
            "best_camera_device_id_to_shelf",
            "best_camera_name_to_shelf",
            "all_in_middle_camera_device_ids_to_shelf",
            "all_in_middle_camera_names_to_shelf",
            "all_in_frame_camera_device_ids_to_shelf",
            "all_in_frame_camera_names_to_shelf",
            "end",
            "description",
            "anonymized_description",
        ]
    )
    material_events.sort_values("timestamp", inplace=True)
    return material_events


def generate_material_events_date_tray(parsed_tray_events_date_tray):
    parsed_tray_events_date_tray_filtered = parsed_tray_events_date_tray.loc[
        parsed_tray_events_date_tray["interaction_type"].isin(["CARRYING_FROM_SHELF", "CARRYING_TO_SHELF"])
    ].sort_values("start")
    in_use = False
    material_events_list = []
    for index, event in parsed_tray_events_date_tray_filtered.iterrows():
        interaction_type = event["interaction_type"]
        if interaction_type == "CARRYING_FROM_SHELF":
            material_events_list.append(
                {
                    "date": event["date"],
                    "tray_device_id": event["tray_device_id"],
                    "material_id": event["material_id"],
                    "material_name": event["material_name"],
                    "start": event["start"],
                    "id_from_shelf": event["id"],
                    "person_device_id_from_shelf": event["person_device_id"],
                    "person_id_from_shelf": event["person_id"],
                    "person_name_from_shelf": event["person_name"],
                    "person_anonymized_name_from_shelf": event["person_anonymized_name"],
                    "person_type_from_shelf": event["person_type"],
                    "best_camera_device_id_from_shelf": event["best_camera_device_id_start"],
                    "best_camera_name_from_shelf": event["best_camera_name_start"],
                    "all_in_middle_camera_device_ids_from_shelf": event["all_in_middle_camera_device_ids_start"],
                    "all_in_middle_camera_names_from_shelf": event["all_in_middle_camera_names_start"],
                    "all_in_frame_camera_device_ids_from_shelf": event["all_in_frame_camera_device_ids_start"],
                    "all_in_frame_camera_names_from_shelf": event["all_in_frame_camera_names_start"],
                    "best_camera_device_id_middle_of_event": None,
                    "best_camera_name_middle_of_event": None,
                    "all_in_middle_camera_device_ids_middle_of_event": None,
                    "all_in_middle_camera_names_middle_of_event": None,
                    "all_in_frame_camera_device_ids_middle_of_event": None,
                    "all_in_frame_camera_names_middle_of_event": None,
                    "id_to_shelf": None,
                    "person_device_id_to_shelf": None,
                    "person_id_to_shelf": None,
                    "person_name_to_shelf": None,
                    "person_anonymized_name_to_shelf": None,
                    "best_camera_device_id_to_shelf": None,
                    "best_camera_name_to_shelf": None,
                    "all_in_middle_camera_device_ids_to_shelf": None,
                    "all_in_middle_camera_names_to_shelf": None,
                    "all_in_frame_camera_device_ids_to_shelf": None,
                    "all_in_frame_camera_names_to_shelf": None,
                    "end": None,
                }
            )
            in_use = True
        elif interaction_type == "CARRYING_TO_SHELF" and in_use:
            material_events_list[-1]["id_to_shelf"] = event["id"]
            material_events_list[-1]["person_device_id_to_shelf"] = event["person_device_id"]
            material_events_list[-1]["person_id_to_shelf"] = event["person_id"]
            material_events_list[-1]["person_name_to_shelf"] = event["person_name"]
            material_events_list[-1]["person_anonymized_name_to_shelf"] = event["person_anonymized_name"]
            material_events_list[-1]["person_type_to_shelf"] = event["person_type"]
            material_events_list[-1]["best_camera_device_id_to_shelf"] = event["best_camera_device_id_end"]
            material_events_list[-1]["best_camera_name_to_shelf"] = event["best_camera_name_end"]
            material_events_list[-1]["all_in_middle_camera_device_ids_to_shelf"] = event[
                "all_in_middle_camera_device_ids_end"
            ]
            material_events_list[-1]["all_in_middle_camera_names_to_shelf"] = (event["all_in_middle_camera_names_end"],)
            material_events_list[-1]["all_in_frame_camera_device_ids_to_shelf"] = (
                event["all_in_frame_camera_device_ids_end"],
            )
            material_events_list[-1]["all_in_frame_camera_names_to_shelf"] = event["all_in_frame_camera_names_end"]
            material_events_list[-1]["end"] = event["end"]
            in_use = False
        elif interaction_type == "CARRYING_TO_SHELF" and not in_use:
            material_events_list.append(
                {
                    "date": event["date"],
                    "tray_device_id": event["tray_device_id"],
                    "material_id": event["material_id"],
                    "material_name": event["material_name"],
                    "start": None,
                    "person_device_id_from_shelf": None,
                    "person_id_from_shelf": None,
                    "all_in_middle_camera_device_ids_from_shelf": None,
                    "all_in_middle_camera_names_from_shelf": None,
                    "all_in_frame_camera_device_ids_from_shelf": None,
                    "all_in_frame_camera_names_from_shelf": None,
                    "person_name_from_shelf": None,
                    "person_anonymized_name_from_shelf": None,
                    "person_type_from_shelf": None,
                    "best_camera_device_id_from_shelf": None,
                    "best_camera_name_from_shelf": None,
                    "id_to_shelf": event["id"],
                    "person_device_id_to_shelf": event["person_device_id"],
                    "person_id_to_shelf": event["person_id"],
                    "person_name_to_shelf": event["person_name"],
                    "person_anonymized_name_to_shelf": event["person_anonymized_name"],
                    "person_type_to_shelf": event["person_type"],
                    "best_camera_device_id_to_shelf": event["best_camera_device_id_end"],
                    "best_camera_name_to_shelf": event["best_camera_name_end"],
                    "all_in_middle_camera_device_ids_to_shelf": event["all_in_middle_camera_device_ids_end"],
                    "all_in_middle_camera_names_to_shelf": event["all_in_middle_camera_names_end"],
                    "all_in_frame_camera_device_ids_to_shelf": event["all_in_frame_camera_device_ids_end"],
                    "all_in_frame_camera_names_to_shelf": event["all_in_frame_camera_names_end"],
                    "end": event["end"],
                }
            )
            in_use = False
        else:
            raise ValueError(
                f"Encountered unexpected state: interaction type is '{interaction_type}' and in_use is {in_use}"
            )
    return material_events_list


def describe_material_event(
    timestamp, material_name, start, person_name_from_shelf, end, person_name_to_shelf, duration_seconds, time_zone
):
    time_string = timestamp.tz_convert(time_zone).strftime("%I:%M %p")
    from_shelf_person_string = person_name_from_shelf if pd.notnull(person_name_from_shelf) else "An unknown person"
    to_shelf_person_string = person_name_to_shelf if pd.notnull(person_name_to_shelf) else "an unknown person"
    if pd.notnull(start) and pd.notnull(end):
        if duration_seconds > 90:
            duration_string = f"{round(duration_seconds / 60)} minutes"
        elif duration_seconds > 30:
            duration_string = "1 minute"
        else:
            duration_string = f"{round(duration_seconds)} seconds"
        if person_name_from_shelf == person_name_to_shelf:
            description_text = "{} took {} from shelf and put it back {} later".format(
                from_shelf_person_string, material_name, duration_string
            )
        else:
            description_text = "{} took {} from shelf and {} put it back {} later".format(
                from_shelf_person_string, material_name, to_shelf_person_string, duration_string
            )
    elif pd.notnull(start):
        description_text = f"{from_shelf_person_string} took {material_name} from shelf but never put it back"
    elif pd.notnull(end):
        if to_shelf_person_string == "an unknown person":
            to_shelf_person_string = to_shelf_person_string.capitalize()
        description_text = "{} put {} back on shelf but it wasn't taken out previously".format(
            to_shelf_person_string, material_name
        )
    else:
        raise ValueError("Unexpected state: both start and end of material event are null")
    description = f"{time_string}: {description_text}"
    return description


def all_cameras_tray_view_data(
    timestamp,
    tray_device_id,
    environment_id=None,
    environment_name=None,
    camera_device_ids=None,
    camera_calibrations=None,
    position_window_seconds=4,
    imputed_z_position=1.0,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
):
    honeycomb_caching_client = HoneycombCachingClient()

    if camera_calibrations is None:
        if environment_id is None and environment_name is None and camera_device_ids is None:
            raise ValueError(
                "If camera calibration info is not specified, must specify either camera device IDs or environment_name ID or environment_name name"
            )
        if camera_device_ids is None:
            camera_info = honeycomb_caching_client.fetch_camera_devices(
                environment_id=environment_id,
                environment_name=environment_name,
                start=timestamp,
                end=timestamp,
                chunk_size=chunk_size,
            )
            camera_device_ids = camera_info.index.unique().tolist()
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_device_ids,
            start=timestamp,
            end=timestamp,
            chunk_size=chunk_size,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret,
        )
    position_window_start = timestamp - datetime.timedelta(seconds=position_window_seconds / 2)
    position_window_end = timestamp + datetime.timedelta(seconds=position_window_seconds / 2)
    position_data = honeycomb_io.fetch_cuwb_position_data(
        start=position_window_start,
        end=position_window_end,
        device_ids=[tray_device_id],
        environment_id=None,
        environment_name=None,
        device_types=["UWBTAG"],
        output_format="dataframe",
        sort_arguments=None,
        chunk_size=1000,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
    )
    position = np.nanmedian(position_data.loc[:, ["x", "y", "z"]].values, axis=0)
    if imputed_z_position is not None:
        position[2] = imputed_z_position
    view_data_list = []
    for camera_device_id, camera_calibration in camera_calibrations.items():
        camera_position = cv_utils.extract_camera_position(
            rotation_vector=camera_calibration["rotation_vector"],
            translation_vector=camera_calibration["translation_vector"],
        )
        distance_from_camera = np.linalg.norm(np.subtract(position, camera_position))
        image_position = cv_utils.project_points(
            object_points=position,
            rotation_vector=camera_calibration["rotation_vector"],
            translation_vector=camera_calibration["translation_vector"],
            camera_matrix=camera_calibration["camera_matrix"],
            distortion_coefficients=camera_calibration["distortion_coefficients"],
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=np.asarray(
                [[0.0, 0.0], [camera_calibration["image_width"], camera_calibration["image_height"]]]
            ),
        )
        image_position = np.squeeze(image_position)
        if np.all(np.isfinite(image_position)):
            in_frame = True
            distance_from_image_center = np.linalg.norm(
                np.subtract(
                    image_position, [camera_calibration["image_width"] / 2, camera_calibration["image_height"] / 2]
                )
            )
            in_middle = (
                image_position[0] > camera_calibration["image_width"] * (1.0 / 10.0)
                and image_position[0] < camera_calibration["image_width"] * (9.0 / 10.0)
                and image_position[1] > camera_calibration["image_height"] * (1.0 / 10.0)
                and image_position[1] < camera_calibration["image_height"] * (9.0 / 10.0)
            )
        else:
            in_frame = False
            distance_from_image_center = None
            in_middle = False
        view_data_list.append(
            {
                "camera_device_id": camera_device_id,
                "position": position,
                "distance_from_camera": distance_from_camera,
                "image_position": image_position,
                "distance_from_image_center": distance_from_image_center,
                "in_frame": in_frame,
                "in_middle": in_middle,
            }
        )
    view_data = pd.DataFrame(view_data_list).set_index("camera_device_id")
    return view_data
