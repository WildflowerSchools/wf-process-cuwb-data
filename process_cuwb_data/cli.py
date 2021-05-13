import itertools
import click
import click_log
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

from .core import estimate_tray_centroids, extract_tray_carry_events_from_inferred, infer_tray_interactions, fetch_cuwb_data, fetch_cuwb_data_from_datapoints, fetch_motion_features, generate_human_activity_groundtruth, generate_human_activity_model, generate_tray_carry_groundtruth, generate_tray_carry_model, infer_human_activity, infer_tray_carry, pose_data_with_body_centroid
from .utils.io import load_csv, read_generic_pkl, write_cuwb_data_pkl, write_datafile_to_csv, write_generic_pkl
from .utils.log import logger
from .uwb_predict_tray_centroids import validate_tray_centroids_dataframe

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
valid_date_formats = list(itertools.chain.from_iterable(
    map(lambda d: ["{}".format(d), "{}%z".format(d), "{} %Z".format(d)], ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'])))


_cli_options_env_start_end = [
    click.option("--environment", type=str, required=True),
    click.option("--start", type=click.DateTime(formats=valid_date_formats), required=True,
                 help="Filter is passed to remote query or used to filter --cuwb-data (if --cuwb-data is provided)"),
    click.option("--end", type=click.DateTime(formats=valid_date_formats), required=True,
                 help="Filter is passed to remote query or used to filter --cuwb-data (if --cuwb-data is provided)")
]

_cli_options_uwb_data = [
    click.option("--cuwb-data", type=click.Path(exists=True), required=False,
                 help="Pickle formatted UWB data (create with 'fetch-cuwb-data')")
]

_cli_options_uwb_motion_data = [
    click.option("--motion-feature-data", type=click.Path(exists=True), required=False,
                 help="Pickle formatted UWB motion data object (create with 'fetch-motion-features')")
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def _load_model_and_scaler(model_path, feature_scaler_path=None):
    model = read_generic_pkl(model_path)

    feature_scaler = None
    if feature_scaler_path is not None:
        feature_scaler = read_generic_pkl(feature_scaler_path)

    return model, feature_scaler


def _load_tray_positions_from_csv(tray_positions_csv):
    df_tray_centroids = None
    if tray_positions_csv is not None:
        try:
            df_tray_centroids = load_csv(tray_positions_csv)
            valid, msg = validate_tray_centroids_dataframe(df_tray_centroids)
            if not valid:
                logger.error(msg)
                return None
        except Exception as err:
            logger.error(err)
            return None

    return df_tray_centroids


def _infer_tray_carry(df_tray_features, model, scaler=None):
    inferred = infer_tray_carry(model=model, scaler=scaler, df_tray_features=df_tray_features)

    df_carry_events = extract_tray_carry_events_from_inferred(inferred)
    if df_carry_events is None or len(df_carry_events) == 0:
        logger.warn("No carry events inferred")
        return None

    return df_carry_events


def _infer_human_activity(df_person_features, model, scaler=None):
    df_person_features_with_nan = df_person_features[df_person_features.isna().any(axis=1)]
    devices_without_acceleration = list(pd.unique(df_person_features_with_nan['device_id']))
    if len(devices_without_acceleration) > 0:
        logger.info("Devices dropped due to missing acceleration data: {}".format(devices_without_acceleration))
        df_person_features.dropna(inplace=True)

    return infer_human_activity(model=model, scaler=scaler, df_person_features=df_person_features)


@click.command(name="fetch-cuwb-data", help="Generate a pickled dataframe of CUWB data")
@add_options(_cli_options_env_start_end)
@click.option("--entity-type", type=click.Choice(['tray', 'person', 'all'],
                                                 case_sensitive=False), default='all', help="CUWB entity type")
@click.option("--data-type", type=click.Choice(['position', 'accelerometer', 'gyroscope', 'magnetometer', 'all'],
                                               case_sensitive=False), default='all', help="Data to return")
@click.option("--data-source", type=click.Choice(['datapoints', 'imu_tables'],
                                                 case_sensitive=False), default='imu_tables', help="Source data resides (datapoints was retired 03/23/2021)")
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder for CUWB data, data stored in <<output>>/uwb_data/<<file>>.pkl")
def cli_fetch_cuwb_data(environment, start, end, entity_type, data_type, data_source, output):
    uwb_output = "{}/uwb_data".format(output)
    Path(uwb_output).mkdir(parents=True, exist_ok=True)

    if data_source == 'datapoints':
        df = fetch_cuwb_data_from_datapoints(environment,
                                             start,
                                             end,
                                             entity_type=entity_type,
                                             data_type=data_type)
    else:
        df = fetch_cuwb_data(
            environment,
            start,
            end,
            entity_type=entity_type,
            data_type=data_type
        )

    if df is None or len(df) == 0:
        logger.warning("No CUWB data found")
        return

    write_cuwb_data_pkl(
        df,
        filename_prefix='uwb',
        environment_name=environment,
        start_time=start,
        end_time=end,
        directory=uwb_output
    )


@click.command(name="fetch-motion-features",
               help="Generate a pickled dataframe of UWB data converted into motion features")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_uwb_data)
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder for cuwb tray features data, features stored in <<output>>/feature_data/<<file>>.pkl")
def cli_fetch_motion_features(environment, start, end, cuwb_data, output):
    feature_data_output = "{}/feature_data".format(output)
    Path(feature_data_output).mkdir(parents=True, exist_ok=True)

    df_uwb_data = None
    if cuwb_data is not None:
        df_uwb_data = read_generic_pkl(cuwb_data)
        df_uwb_data = df_uwb_data.loc[(df_uwb_data.index >= start) & (df_uwb_data.index <= end)]

    df_features = fetch_motion_features(
        environment,
        start,
        end,
        include_meta_fields=True,
        df_uwb_data=df_uwb_data
    )

    if df_features is None or len(df_features) == 0:
        logger.warning("No CUWB data found")
        return

    write_cuwb_data_pkl(
        df_features,
        filename_prefix='motion-features',
        environment_name=environment,
        start_time=start,
        end_time=end,
        directory=feature_data_output
    )


@click.command(name="generate-tray-carry-groundtruth",
               help="Generate a pickled dataframe of trainable groundtruth features")
@click.option("--groundtruth-csv", type=click.Path(exists=True),
              help="CSV formatted groundtruth data", required=True)
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, output includes data features pickle (<<output>>/groundtruth/<<now>>_tray_carry_groundtruth_features.pkl)")
def cli_generate_tray_carry_groundtruth(groundtruth_csv, output):
    groundtruth_features_output = "{}/groundtruth".format(output)
    Path(groundtruth_features_output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = generate_tray_carry_groundtruth(groundtruth_csv)

    if df_groundtruth_features is None:
        logger.warn("Unexpected result, unable to store groundtruth features")
    else:
        write_generic_pkl(
            df_groundtruth_features,
            "{}_tray_carry_groundtruth_features".format(now),
            groundtruth_features_output)


@click.command(name="generate-human-activity-groundtruth",
               help="Generate a pickled dataframe of trainable groundtruth features")
@click.option("--groundtruth-csv", type=click.Path(exists=True),
              help="CSV formatted groundtruth data", required=True)
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, output includes data features pickle (<<output>>/groundtruth/<<now>>_human_activity_groundtruth_features.pkl)")
def cli_generate_human_activity_groundtruth(groundtruth_csv, output):
    groundtruth_features_output = "{}/groundtruth".format(output)
    Path(groundtruth_features_output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = generate_human_activity_groundtruth(groundtruth_csv)

    if df_groundtruth_features is None:
        logger.warn("Unexpected result, unable to store groundtruth features")
    else:
        write_generic_pkl(
            df_groundtruth_features,
            "{}_human_activity_groundtruth_features".format(now),
            groundtruth_features_output)


@click.command(name="train-human-activity-model",
               help="Train and generate a pickled model and feature scaler given groundtruth features")
@click.option("--groundtruth-features", type=click.Path(exists=True),
              help="Pickle formatted groundtruth features data (create with 'generate-human-activity-groundtruth')")
@click.option("--output", type=click.Path(), default="%s/output/models" % (os.getcwd()),
              help="output folder, model output includes pickled model (<<output>>/models/<<DATE>>_model.pkl) and pickled scaler (<<output>>/models/<<DATE>>_scaler.pkl)")
def cli_train_human_activity_model(groundtruth_features, output):
    models_output = "{}/models".format(output)
    Path(models_output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = pd.read_pickle(groundtruth_features)
    result = generate_human_activity_model(df_groundtruth_features)

    if result is not None:
        write_generic_pkl(result['model'], "{}_tray_carry_model".format(now), models_output)

        if result['scaler'] is not None:
            write_generic_pkl(result['scaler'], "{}_tray_carry_scaler".format(now), models_output)


@click.command(name="train-tray-carry-model",
               help="Train and generate a pickled model and feature scaler given groundtruth features")
@click.option("--groundtruth-features", type=click.Path(exists=True),
              help="Pickle formatted groundtruth features data (create with 'generate-tray-carry-groundtruth')")
@click.option("--tune", is_flag=True,
              default=False, help="Tune the classifier, yields ideal hyperparameters")
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, model output includes pickled model (<<output>>/models/<<DATE>>_model.pkl) and pickled scaler (<<output>>/models/<<DATE>>_scaler.pkl).")
def cli_train_tray_carry_model(groundtruth_features, tune, output):
    models_output = "{}/models".format(output)
    Path(models_output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = pd.read_pickle(groundtruth_features)
    result = generate_tray_carry_model(df_groundtruth_features, tune=tune)

    if result is not None:
        write_generic_pkl(result['model'], "{}_human_activity_model".format(now), models_output)

        if result['scaler'] is not None:
            write_generic_pkl(result['scaler'], "{}_human_activity_scaler".format(now), models_output)


@click.command(name="estimate-tray-centroids",
               help="Estimate tray shelf locations. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_uwb_data)
@add_options(_cli_options_uwb_motion_data)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, tray centroids stored as csv in <<output>>/locations (e.g. <<output/locations/<<DATE>>_tray_centroids.csv)")
def cli_estimate_tray_centroids(environment, start, end, cuwb_data, motion_feature_data, model, feature_scaler, output):
    locations_output = "{}/locations".format(output)
    Path(locations_output).mkdir(parents=True, exist_ok=True)

    df_uwb_data = None
    if cuwb_data is not None:
        df_uwb_data = read_generic_pkl(cuwb_data)
        df_uwb_data = df_uwb_data.loc[(df_uwb_data.index >= start) & (df_uwb_data.index <= end)]

    if motion_feature_data is None:
        df_uwb_motion_features = fetch_motion_features(
            environment,
            start,
            end,
            include_meta_fields=True,
            df_uwb_data=df_uwb_data,
            fillna='interpolate'
        )
    else:
        df_uwb_motion_features = read_generic_pkl(motion_feature_data)
        df_uwb_motion_features = df_uwb_motion_features.loc[(
            df_uwb_motion_features.index >= start) & (df_uwb_motion_features.index <= end)]

    df_tray_features = df_uwb_motion_features[df_uwb_motion_features['entity_type'] == "Tray"]
    model_obj, feature_scaler_obj = _load_model_and_scaler(model_path=model, feature_scaler_path=feature_scaler)

    df_tray_centroids = estimate_tray_centroids(
        model=model_obj,
        scaler=feature_scaler_obj,
        df_tray_features=df_tray_features)
    if df_tray_centroids is None or len(df_tray_centroids) == 0:
        logger.warn("No tray centroids inferred")
        return
    else:
        write_datafile_to_csv(df_tray_centroids,
                              "{}_tray_centroids".format(now),
                              directory=locations_output,
                              index=False)


@click.command(name="infer-tray-carry",
               help="Infer tray carrying events given a model and feature scaler. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_uwb_data)
@add_options(_cli_options_uwb_motion_data)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, carry events stored as csv in <<output>>/inference/tray_carry (e.g. <<output>>/inference/tray_carry/<<DATE>>_carry_events.csv)")
def cli_infer_tray_carry(environment, start, end, cuwb_data, motion_feature_data, model, feature_scaler, output):
    inference_output = "{}/inference/tray_carry".format(output)
    Path(inference_output).mkdir(parents=True, exist_ok=True)

    df_uwb_data = None
    if cuwb_data is not None:
        df_uwb_data = read_generic_pkl(cuwb_data)
        df_uwb_data = df_uwb_data.loc[(df_uwb_data.index >= start) & (df_uwb_data.index <= end)]

    if motion_feature_data is None:
        df_uwb_motion_features = fetch_motion_features(
            environment,
            start,
            end,
            df_uwb_data=df_uwb_data,
            fillna='interpolate'
        )
    else:
        df_uwb_motion_features = read_generic_pkl(motion_feature_data)
        df_uwb_motion_features = df_uwb_motion_features.loc[(
            df_uwb_motion_features.index >= start) & (df_uwb_motion_features.index <= end)]

    df_tray_features = df_uwb_motion_features[df_uwb_motion_features['entity_type'] == "Tray"]

    if df_tray_features is None or len(df_tray_features) == 0:
        logger.warn("No tray motion events detected")
        return None

    model_obj, feature_scaler_obj = _load_model_and_scaler(model, feature_scaler)
    df_carry_events = _infer_tray_carry(
        df_tray_features=df_tray_features,
        model=model_obj,
        scaler=feature_scaler_obj)
    if df_carry_events is None or len(df_carry_events) == 0:
        logger.warn("No tray carry events detected")
        return

    write_datafile_to_csv(df_carry_events,
                          "{}_carry_events".format(now),
                          directory=inference_output,
                          index=False)


@click.command(name="infer-human-activity",
               help="Infer human carrying events given a model and feature scaler. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_uwb_motion_data)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-human-activity-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-human-activity-model')")
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, carry events stored as csv in <<output>>/inference/human_activity (e.g. <<output>>/inference/human_activity/<<DATE>>_human_activity.csv)")
def cli_infer_human_activity(environment,
                             start,
                             end,
                             motion_feature_data,
                             model,
                             feature_scaler,
                             output):
    inference_output = "{}/inference/human_activity".format(output)
    Path(inference_output).mkdir(parents=True, exist_ok=True)

    if motion_feature_data is None:
        df_uwb_motion_features = fetch_motion_features(
            environment,
            start,
            end
        )
    else:
        df_uwb_motion_features = read_generic_pkl(motion_feature_data)
        df_uwb_motion_features = df_uwb_motion_features.loc[(
            df_uwb_motion_features.index >= start) & (df_uwb_motion_features.index <= end)]

    df_uwb_motion_features = df_uwb_motion_features.interpolate().fillna(method='bfill')

    df_person_features = df_uwb_motion_features[df_uwb_motion_features['entity_type'] == "Person"]
    if df_person_features is None or len(df_person_features) == 0:
        logger.warn("No person motion events detected")
        return

    model_obj, feature_scaler_obj = _load_model_and_scaler(model, feature_scaler)
    df_person_features_with_har = _infer_human_activity(
        df_person_features=df_person_features,
        model=model_obj,
        scaler=feature_scaler_obj)
    if df_person_features_with_har is None or len(df_person_features_with_har) == 0:
        logger.warn("No human activity detected")
        return

    write_datafile_to_csv(df_person_features_with_har,
                          "{}_human_activity".format(now),
                          directory=inference_output,
                          index=False)


@click.command(name="infer-tray-interactions",
               help="Infer tray interactions (CARRY FROM SHELF / CARRY TO SHELF / CARRY UNKOWN / etc.) given a model and feature scaler. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_uwb_data)
@add_options(_cli_options_uwb_motion_data)
@click.option("--tray-carry-model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--tray-carry-feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--human-activity-model", type=click.Path(exists=True),
              help="Pickle formatted model object (create with 'human-activity-carry-model')")
@click.option("--human-activity-feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'human-activity-carry-model')")
@click.option("--pose-inference-id", type=str,
              help="3D Pose Inference ID (requires and searches <<output>>/pose_processing folder)")
@click.option("--pose-inference-base", type=click.Path(exists=True),
              help="3D Pose Inference directory base path")
@click.option("--pose-inference-subdirectory", type=str,
              help="3D Pose Inference subdirectory folder", default="pose_processing")
@click.option("--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, tray interactions stored as csv in <<output>>/inference/tray_interactions (e.g. <<output>>/inference/interactions/<<DATE>>_tray_interactions.csv)")
@click.option("--tray-positions-csv", type=click.Path(exists=True),
              help="CSV formatted tray shelf position data")
def cli_infer_tray_interactions(environment,
                                start,
                                end,
                                cuwb_data,
                                motion_feature_data,
                                tray_carry_model,
                                tray_carry_feature_scaler,
                                human_activity_model,
                                human_activity_feature_scaler,
                                pose_inference_id,
                                pose_inference_base,
                                pose_inference_subdirectory,
                                output,
                                tray_positions_csv):
    interactions_output = "{}/inference/tray_interactions".format(output)
    Path(interactions_output).mkdir(parents=True, exist_ok=True)

    if pose_inference_base is None:
        pose_inference_base = output

    Path("{}/{}".format(pose_inference_base, pose_inference_subdirectory)).mkdir(parents=True, exist_ok=True)

    if motion_feature_data is None:
        if cuwb_data is not None:
            df_uwb_data = read_generic_pkl(cuwb_data)
            df_uwb_data = df_uwb_data.loc[(df_uwb_data.index >= start) & (df_uwb_data.index <= end)]
        else:
            df_uwb_data = fetch_cuwb_data(
                environment,
                start,
                end
            )

        df_uwb_motion_features = fetch_motion_features(
            environment,
            start,
            end,
            df_uwb_data=df_uwb_data,
            fillna='interpolate'
        )
    else:
        df_uwb_motion_features = read_generic_pkl(motion_feature_data)
        df_uwb_motion_features = df_uwb_motion_features.loc[(
            df_uwb_motion_features.index >= start) & (df_uwb_motion_features.index <= end)]

    df_person_features = df_uwb_motion_features[df_uwb_motion_features['entity_type'] == "Person"]
    # For human activity predictions
    if df_person_features is None or len(df_person_features) == 0:
        logger.warn("No person motion events detected")
        return

    # For tray carry predictions
    df_tray_features = df_uwb_motion_features[df_uwb_motion_features['entity_type'] == "Tray"]
    if df_tray_features is None or len(df_tray_features) == 0:
        logger.warn("No tray motion events detected")
        return

    # human_activity_model_obj, human_activity_feature_scaler_obj = _load_model_and_scaler(
    #     human_activity_model, human_activity_feature_scaler)
    # df_person_features_with_har = _infer_human_activity(
    #     df_person_features=df_person_features,
    #     model=human_activity_model_obj,
    #     scaler=human_activity_feature_scaler_obj)

    tray_carry_model_obj, tray_carry_feature_scaler_obj = _load_model_and_scaler(
        tray_carry_model, tray_carry_feature_scaler)
    df_carry_events = _infer_tray_carry(
        df_tray_features=df_tray_features,
        model=tray_carry_model_obj,
        scaler=tray_carry_feature_scaler_obj)
    if df_carry_events is None or len(df_carry_events) == 0:
        logger.warn("No tray carry events detected")
        return

    if tray_positions_csv is not None:
        df_tray_centroids = _load_tray_positions_from_csv(tray_positions_csv)
    else:
        df_tray_centroids = estimate_tray_centroids(
            model=tray_carry_model_obj,
            scaler=tray_carry_feature_scaler_obj,
            df_tray_features=df_tray_features)

    if df_tray_centroids is None or len(df_tray_centroids) == 0:
        logger.warn("No tray centroids inferred")
        return

    df_poses_3d = None
    if pose_inference_id is not None:
        df_poses_3d = pose_data_with_body_centroid(environment=environment,
                                                   start=start,
                                                   end=end,
                                                   pose_inference_id=pose_inference_id,
                                                   pose_inference_base=pose_inference_base,
                                                   pose_inference_subdirectory=pose_inference_subdirectory)

    # filter by person_id (note nan, perhaps we filter our known tracks w/ person_id?)
    # convert 'keypoint_coordinates_3d' to 'position_x/y/z'

    # # Append the Person Activity labels to the motion features dataframe, name column "human_activity_category"
    # df_all_motion_features = pd.merge(df_uwb_motion_features.reset_index(),
    #                                   df_person_features_with_har[['device_id',
    #                                                                'predicted_human_activity_label']].reset_index(),
    #                                   how='left',
    #                                   on=['index', 'device_id']) \
    #     .set_index('index') \
    #     .rename(columns={'predicted_human_activity_label': 'human_activity_category'})
    df_tray_interactions = infer_tray_interactions(df_motion_features=df_uwb_motion_features.copy(),
                                                   df_carry_events=df_carry_events,
                                                   df_tray_centroids=df_tray_centroids,
                                                   df_poses_3d=df_poses_3d)
    if df_tray_interactions is None or len(df_tray_interactions) == 0:
        logger.warn("No tray interactions inferred")
        return
    else:
        write_datafile_to_csv(df_tray_interactions,
                              "{}_tray_interactions".format(now),
                              directory=interactions_output,
                              index=False)


@click_log.simple_verbosity_option(logger)
@click.group()
@click.option("--env-file", type=click.Path(exists=True),
              help="env file to load environment variables from")
def cli(env_file):
    if env_file is None:
        env_file = os.path.join(os.getcwd(), '.env')

    if os.path.exists(env_file):
        load_dotenv(dotenv_path=env_file)


cli.add_command(cli_fetch_motion_features)
cli.add_command(cli_fetch_cuwb_data)
cli.add_command(cli_generate_human_activity_groundtruth)
cli.add_command(cli_generate_tray_carry_groundtruth)
cli.add_command(cli_train_human_activity_model)
cli.add_command(cli_train_tray_carry_model)
cli.add_command(cli_estimate_tray_centroids)
cli.add_command(cli_infer_tray_carry)
cli.add_command(cli_infer_human_activity)
cli.add_command(cli_infer_tray_interactions)
