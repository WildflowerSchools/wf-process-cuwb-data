from datetime import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

import process_cuwb_data
from process_cuwb_data.utils import io

import dotenv

from process_cuwb_data.utils.util import filter_by_data_type
from process_cuwb_data.uwb_motion_enum_carry_categories import CarryCategory

dotenv.load_dotenv("../.env")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def run(environment_name, start, end, models):
    df_uwb_data = process_cuwb_data.fetch_cuwb_data(
        environment_name=environment_name, start=start, end=end, overwrite_cache=True
    )

    df_uwb_motion_features = process_cuwb_data.fetch_motion_features(
        environment_name=environment_name,
        start=start,
        end=end,
        df_uwb_data=df_uwb_data,
        fillna="forward_backward",
        overwrite_cache=True,
    )

    all_carry_events = []
    all_inferred_tray_carries = []
    for model in models:
        if model["device_part_number"] is None:
            uwb_device_filter = "All"
            df_uwb_motion_features_for_model = df_uwb_motion_features.copy()
        else:
            uwb_device_filter = model["device_part_number"]
            df_uwb_motion_features_for_model = df_uwb_motion_features[
                df_uwb_motion_features["device_part_number"].str.lower() == model["device_part_number"].lower()
            ]

        logging.info(f"Running inference against '{uwb_device_filter}' UWB devices")

        df_inferred_tray_carry_for_model = process_cuwb_data.infer_tray_carry(
            df_tray_features=df_uwb_motion_features_for_model, model=model["model"]
        )

        df_carry_events_for_model = process_cuwb_data.extract_tray_carry_events_from_inferred(
            df_inferred=df_inferred_tray_carry_for_model
        )

        all_carry_events.append(df_carry_events_for_model)
        all_inferred_tray_carries.append(df_inferred_tray_carry_for_model)

    df_carry_events = pd.concat(all_carry_events)
    df_inferred_tray_carry = pd.concat(all_inferred_tray_carries)

    df_tray_features_not_carried = df_inferred_tray_carry[
        df_inferred_tray_carry["predicted_tray_carry_label"] == CarryCategory.NOT_CARRIED.name
    ]

    df_tray_centroids = process_cuwb_data.estimate_tray_centroids(
        environment_name=environment_name,
        start=start,
        end=end,
        df_tray_features_not_carried=df_tray_features_not_carried,
        overwrite_cache=True,
    )

    df_tray_interactions = process_cuwb_data.infer_tray_interactions(
        df_motion_features=df_uwb_motion_features, df_carry_events=df_carry_events, df_tray_centroids=df_tray_centroids
    )

    # Turn tray interactions into "events"
    df_tray_events = process_cuwb_data.infer_tray_events(
        df_tray_interactions=df_tray_interactions,
        environment_name=environment_name,
        time_zone="US/Pacific",
        df_cuwb_position_data=filter_by_data_type(df_uwb_data, "position"),
    )

    df_material_events = process_cuwb_data.generate_material_events(
        df_parsed_tray_events=df_tray_events,
        environment_name=environment_name,
        time_zone="US/Pacific",
        df_cuwb_position_data=filter_by_data_type(df_uwb_data, "position"),
    )

    inference_output_path_tray_events = "../output/inference/tray_events"
    Path(inference_output_path_tray_events).mkdir(parents=True, exist_ok=True)

    inference_output_path_material_events = "../output/inference/material_events"
    Path(inference_output_path_material_events).mkdir(parents=True, exist_ok=True)

    csv_tray_events_output_path = (
        f"{inference_output_path_tray_events}/tray_events_{environment_name}_{start.strftime('%Y_%m_%d')}.csv"
    )
    df_tray_events.to_csv(csv_tray_events_output_path, index=False)
    logging.info(f"Wrote {len(df_tray_events)} df_tray_events events to {csv_tray_events_output_path}")

    csv_material_events_output_path = (
        f"{inference_output_path_material_events}/material_events_{environment_name}_{start.strftime('%Y_%m_%d')}.csv"
    )
    df_material_events.to_csv(csv_material_events_output_path, index=False)
    logging.info(f"Wrote {len(df_material_events)} material events to {csv_material_events_output_path}")


if __name__ == "__main__":
    tray_detection_model_v2_path = "../output/models/tray_carry_model_v2.pkl"
    tray_detection_model_dwtag100_path = "../output/models/dwtag100_tray_carry_model.pkl"
    tray_detection_model_pt202_path = "../output/models/pt202_tray_carry_model.pkl"

    models = [
        # {
        #     "model": io.read_generic_pkl(tray_detection_model_v2_path),
        #     "device_part_number": None
        # }
        {"model": io.read_generic_pkl(tray_detection_model_dwtag100_path), "device_part_number": "dwtag100"},
        {
            "model": io.read_generic_pkl(tray_detection_model_pt202_path),
            "device_part_number": "pt202",
        },
    ]

    environment_name = "dahlia"
    # start = datetime.strptime("2023-07-20T07:30:00-0800", "%Y-%m-%dT%H:%M:%S%z")
    # end = datetime.strptime("2023-07-20T17:30:00-0800", "%Y-%m-%dT%H:%M:%S%z")
    # run(environment_name, start, end, models)

    start = datetime.strptime("2023-08-28T08:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    end = datetime.strptime("2023-08-28T18:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    run(environment_name, start, end, models)

    # start = datetime.strptime("2023-08-29T08:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    # end = datetime.strptime("2023-08-29T18:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    # run(environment_name, start, end, models)

    # start = datetime.strptime("2023-08-31T08:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    # end = datetime.strptime("2023-08-31T18:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    # run(environment_name, start, end, models)

    # start = datetime.strptime("2023-09-07T08:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    # end = datetime.strptime("2023-09-07T18:30:00-0700", "%Y-%m-%dT%H:%M:%S%z")
    # run(environment_name, start, end, models)
