import logging
import sys

import numpy as np
import pandas as pd

import process_cuwb_data

import dotenv

dotenv.load_dotenv(".env")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def run(environment_name, start, end, df_carried_groundtruth):
    df_uwb_data = process_cuwb_data.fetch_cuwb_data(environment_name=environment_name, start=start, end=end)

    df_uwb_motion_tray_features = process_cuwb_data.fetch_motion_features(
        environment_name=environment_name,
        start=start,
        end=end,
        df_uwb_data=df_uwb_data,
        fillna="forward_backward",
        entity_type="tray",
    )

    periods_with_no_inferred_carry = []
    df_uwb_motion_tray_features = df_uwb_motion_tray_features.sort_index()

    for device_id, df_uwb_motion_features_for_device in df_uwb_motion_tray_features.groupby("device_id"):
        df_uwb_motion_features_for_device["active_session_id"] = (
            df_uwb_motion_features_for_device.index.to_series().diff() >= pd.to_timedelta("10 seconds")
        ).cumsum()
        for _, df_uwb_motion_features_for_device_by_session in df_uwb_motion_features_for_device.groupby(
            by="active_session_id"
        ):
            df_uwb_motion_features_for_device_by_session = df_uwb_motion_features_for_device_by_session.copy()

            all_masks = []
            for idx, row_groundtruth in df_carried_groundtruth.iterrows():
                if row_groundtruth["device_id"] == device_id:
                    mask = (df_uwb_motion_features_for_device_by_session.index >= row_groundtruth["start_time"]) & (
                        df_uwb_motion_features_for_device_by_session.index <= row_groundtruth["end_time"]
                    )
                    all_masks.append(mask)

            if len(all_masks) > 0:
                df_uwb_motion_features_for_device_by_session_filtered = (
                    df_uwb_motion_features_for_device_by_session.loc[~np.logical_or.reduce(all_masks)].copy()
                )
            else:
                df_uwb_motion_features_for_device_by_session_filtered = (
                    df_uwb_motion_features_for_device_by_session.copy()
                )

            logging.info(
                f"Extracting groundtruth from {environment_name} for tag '{device_id}' between {start} and {end}. Session contained {len(df_uwb_motion_features_for_device_by_session)} moments, after filtering session included {len(df_uwb_motion_features_for_device_by_session_filtered)} moments"
            )

            df_uwb_motion_features_for_device_by_session_filtered["filtered_session_id"] = (
                df_uwb_motion_features_for_device_by_session_filtered.index.to_series().diff()
                >= pd.to_timedelta("1 seconds")
            ).cumsum()
            for (
                _,
                df_uwb_motion_features_for_device_by_session_filtered_session,
            ) in df_uwb_motion_features_for_device_by_session_filtered.groupby(by="filtered_session_id"):
                periods_with_no_inferred_carry.append(
                    {
                        "environment": environment_name,
                        "start_time": df_uwb_motion_features_for_device_by_session_filtered_session.index.min(),
                        "end_time": df_uwb_motion_features_for_device_by_session_filtered_session.index.max(),
                        "device_id": device_id,
                        "tag_type": df_uwb_motion_features_for_device_by_session["device_part_number"]
                        .unique()[0]
                        .lower(),
                        "device_serial_number": df_uwb_motion_features_for_device_by_session[
                            "device_serial_number"
                        ].unique()[0],
                        "wos_enabled": True,
                        "material_description": df_uwb_motion_features_for_device_by_session["material_name"].unique()[
                            0
                        ],
                        "person_description": "",
                        "description_of_activity": "",
                        "ground_truth_state": "Not Carried",
                        "source": "Identified active tag moments and filtered out Carried moments - generated 9/14/2023",
                    }
                )

    return pd.DataFrame(periods_with_no_inferred_carry)


if __name__ == "__main__":
    environment_name = "dahlia"

    df_grouth_truth_all = pd.read_csv("./groundtruth/ground_truth_tray_carry_all.csv")
    df_grouth_truth_all["start_time"] = pd.to_datetime(
        df_grouth_truth_all["start_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f%z"
    ).fillna(pd.to_datetime(df_grouth_truth_all["start_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S%z"))
    df_grouth_truth_all["end_time"] = pd.to_datetime(
        df_grouth_truth_all["end_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f%z"
    ).fillna(pd.to_datetime(df_grouth_truth_all["end_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S%z"))

    df_grouth_truth_carried_dahlia = df_grouth_truth_all[
        (df_grouth_truth_all["environment"] == "dahlia") & (df_grouth_truth_all["ground_truth_state"] == "Carried")
    ]

    all_not_carried_groundtruth = []
    for date, df_grouth_truth_carried_dahlia_for_day in df_grouth_truth_carried_dahlia.groupby(
        df_grouth_truth_carried_dahlia["start_time"].dt.date
    ):
        df_output = run(
            environment_name=environment_name,
            start=pd.to_datetime(date).tz_localize("US/Pacific").replace(hour=7, minute=30),
            end=pd.to_datetime(date).tz_localize("US/Pacific").replace(hour=16, minute=30),
            df_carried_groundtruth=df_grouth_truth_carried_dahlia_for_day,
        )
        all_not_carried_groundtruth.append(df_output)

    df_all_not_carried_groundtruth = pd.concat(all_not_carried_groundtruth)

    output_path = f"./output/no_carry/activities_with_no_inferred_carry.csv"
    logging.info(f"Outputting activities_with_no_inferred_carry to '{output_path}'")
    df_all_not_carried_groundtruth.to_csv(index=False, path_or_buf=output_path)
