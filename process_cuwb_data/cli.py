import itertools
import click
import click_log
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

from .core import estimate_tray_centroids, extract_motion_features_from_raw, extract_tray_carry_events_from_inferred, extract_tray_interactions, fetch_cuwb_data, fetch_motion_features, generate_human_activity_groundtruth, generate_human_activity_model, generate_tray_carry_groundtruth, generate_tray_carry_model, infer_human_activity, infer_tray_carry
from process_cuwb_data.utils.io import load_csv, read_generic_pkl, write_cuwb_data_pkl, write_datafile_to_csv, write_generic_pkl
from process_cuwb_data.utils.log import logger
from .uwb_predict_tray_centroids import validate_tray_centroids_dataframe

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
valid_date_formats = list(itertools.chain.from_iterable(
    map(lambda d: ["{}".format(d), "{}%z".format(d), "{} %Z".format(d)], ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'])))


_cli_options_env_start_end = [
    click.option("--environment", type=str, required=True),
    click.option("--start", type=click.DateTime(formats=valid_date_formats), required=True, help="Filter is passed to remote query or used to filter --cuwb-data (if --cuwb-data is provided)"),
    click.option("--end", type=click.DateTime(formats=valid_date_formats), required=True, help="Filter is passed to remote query or used to filter --cuwb-data (if --cuwb-data is provided)")
]

_cli_options_cuwb = [
    click.option("--cuwb-data", type=click.Path(exists=True), required=False,
                 help="Pickle formatted cuwb data object (create with 'fetch-cuwb-data'). Avoids remotely querying Honeycomb by overriding environment/start/end inputs.")
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


@click.command(name="fetch-cuwb-data", help="Generate a pickled dataframe of cuwb data")
@add_options(_cli_options_env_start_end)
@click.option("--entity-type", type=click.Choice(['tray', 'person', 'all'],
                                                 case_sensitive=False), default='all', help="CUWB entity type")
@click.option("--data-type", type=click.Choice(['position', 'accelerometer', 'raw'],
                                               case_sensitive=False), default='raw', help="Data to return")
@click.option("--environment-assignments/--no-environment-assignments", is_flag=True,
              default=False, help="Show assignment IDs in addition to device IDs")
@click.option("--entity-assignments/--no-entity-assignments", is_flag=True,
              default=False, help="Map CUWB device IDs to specific trays and people")
@click.option("--output", type=click.Path(), default="%s/output/data" % (os.getcwd()),
              help="output folder for cuwb data")
def cli_fetch_cuwb_data(environment, start, end, entity_type, data_type,
                        environment_assignments, entity_assignments, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df = fetch_cuwb_data(
        environment,
        start,
        end,
        entity_type=entity_type,
        data_type=data_type,
        environment_assignment_info=environment_assignments,
        entity_assignment_info=entity_assignments
    )

    if df is None or len(df) == 0:
        logger.warning("No CUWB data found")
        return

    write_cuwb_data_pkl(
        df,
        filename_prefix='',
        environment_name=environment,
        start_time=start,
        end_time=end,
        environment_assignment_info=environment_assignments,
        entity_assignment_info=entity_assignments,
        directory=output
    )


@click.command(name="generate-tray-carry-groundtruth",
               help="Generate a pickled dataframe of trainable groundtruth features")
@click.option("--groundtruth-csv", type=click.Path(exists=True),
              help="CSV formatted groundtruth data", required=True)
@click.option("--output", type=click.Path(), default="%s/output/features" % (os.getcwd()),
              help="output folder, output includes data features pickle (features.pkl)")
def cli_generate_tray_carry_groundtruth(groundtruth_csv, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = generate_tray_carry_groundtruth(groundtruth_csv)

    if df_groundtruth_features is None:
        logger.warn("Unexpected result, unable to store groundtruth features")
    else:
        write_generic_pkl(df_groundtruth_features, "{}_tray_carry_features".format(now), output)


@click.command(name="generate-human-activity-groundtruth",
               help="Generate a pickled dataframe of trainable groundtruth features")
@click.option("--groundtruth-csv", type=click.Path(exists=True),
              help="CSV formatted groundtruth data", required=True)
@click.option("--output", type=click.Path(), default="%s/output/features" % (os.getcwd()),
              help="output folder, output includes data features pickle (features.pkl)")
def cli_generate_human_activity_groundtruth(groundtruth_csv, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = generate_human_activity_groundtruth(groundtruth_csv)

    if df_groundtruth_features is None:
        logger.warn("Unexpected result, unable to store groundtruth features")
    else:
        write_generic_pkl(df_groundtruth_features, "{}_human_activity_features".format(now), output)


@click.command(name="train-human-activity-model",
               help="Train and generate a pickled model and feature scaler given groundtruth features")
@click.option("--groundtruth-features", type=click.Path(exists=True),
              help="Pickle formatted groundtruth features data (create with 'generate-human-activity-groundtruth')")
@click.option("--output", type=click.Path(), default="%s/output/models" % (os.getcwd()),
              help="output folder, model output includes pickled model (<<DATE>>_model.pkl) and pickled scaler (<<DATE>>_scaler.pkl)")
def cli_train_human_activity_model(groundtruth_features, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = pd.read_pickle(groundtruth_features)
    result = generate_human_activity_model(df_groundtruth_features)

    if result is not None:
        write_generic_pkl(result['model'], "{}_model".format(now), output)

        if result['scaler'] is not None:
            write_generic_pkl(result['scaler'], "{}_scaler".format(now), output)


@click.command(name="train-tray-carry-model",
               help="Train and generate a pickled model and feature scaler given groundtruth features")
@click.option("--groundtruth-features", type=click.Path(exists=True),
              help="Pickle formatted groundtruth features data (create with 'generate-tray-carry-groundtruth')")
@click.option("--tune", is_flag=True,
              default=False, help="Tune the classifier, yields ideal hyperparameters")
@click.option("--output", type=click.Path(), default="%s/output/models" % (os.getcwd()),
              help="output folder, model output includes pickled model (<<DATE>>_model.pkl) and pickled scaler (<<DATE>>_scaler.pkl)")
def cli_train_tray_carry_model(groundtruth_features, tune, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = pd.read_pickle(groundtruth_features)
    result = generate_tray_carry_model(df_groundtruth_features, tune=tune)

    if result is not None:
        write_generic_pkl(result['model'], "{}_model".format(now), output)

        if result['scaler'] is not None:
            write_generic_pkl(result['scaler'], "{}_scaler".format(now), output)


@click.command(name="estimate-tray-centroids",
               help="Estimate tray shelf locations. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_cuwb)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--output", type=click.Path(), default="%s/output/locations" % (os.getcwd()),
              help="output folder, tray centroids as csv (<<DATE>>_tray_centroids.csv)")
def cli_estimate_tray_centroids(environment, start, end, cuwb_data, model, feature_scaler, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    if cuwb_data is None:
        df_cuwb_features = fetch_cuwb_data(
            environment,
            start,
            end,
            environment_assignment_info=True,
            entity_assignment_info=True
        )
    else:
        df_cuwb_features = read_generic_pkl(cuwb_data)
        df_cuwb_features = df_cuwb_features.loc[(df_cuwb_features.index >= start) & (df_cuwb_features.index <= end)]

    df_tray_features = extract_motion_features_from_raw(
        df_cuwb_features=df_cuwb_features,
        entity_type='tray',
        include_meta_fields=True,
        fillna='interpolate')

    model_obj, feature_scaler_obj = _load_model_and_scaler(model_path=model, feature_scaler_path=feature_scaler)

    df_tray_centroids = estimate_tray_centroids(
        model=model_obj,
        scaler=feature_scaler_obj,
        df_tray_features=df_tray_features)
    if df_tray_centroids is None or len(df_tray_centroids) == 0:
        logger.warn("No tray centroids inferred")
        return
    else:
        write_datafile_to_csv(df_tray_centroids, "{}_tray_centroids".format(now), directory=output, index=False)


@click.command(name="infer-tray-carry",
               help="Infer tray carrying events given a model and feature scaler. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_cuwb)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--output", type=click.Path(), default="%s/output/inference" % (os.getcwd()),
              help="output folder, carry events as csv (<<DATE>>_carry_events.csv)")
def cli_infer_tray_carry(environment, start, end, cuwb_data, model, feature_scaler, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    if cuwb_data is None:
        df_cuwb_features = fetch_cuwb_data(
            environment,
            start,
            end,
            environment_assignment_info=True,
            entity_assignment_info=True
        )
    else:
        df_cuwb_features = read_generic_pkl(cuwb_data)
        df_cuwb_features = df_cuwb_features.loc[(df_cuwb_features.index >= start) & (df_cuwb_features.index <= end)]

    df_tray_features = extract_motion_features_from_raw(
        df_cuwb_features=df_cuwb_features,
        entity_type='tray',
        include_meta_fields=True,
        fillna='interpolate')

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

    write_datafile_to_csv(df_carry_events, "{}_carry_events".format(now), directory=output, index=False)


@click.command(name="infer-human-activity",
               help="Infer human carrying events given a model and feature scaler. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_cuwb)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-human-activity-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-human-activity-model')")
@click.option("--output", type=click.Path(), default="%s/output/inference" % (os.getcwd()),
              help="output folder, carry events as csv (<<DATE>>_carry_events.csv)")
def cli_infer_human_activity(environment, start, end, cuwb_data, model, feature_scaler, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    if cuwb_data is None:
        df_cuwb_features = fetch_cuwb_data(
            environment,
            start,
            end,
            environment_assignment_info=True,
            entity_assignment_info=True
        )
    else:
        df_cuwb_features = read_generic_pkl(cuwb_data)
        df_cuwb_features = df_cuwb_features.loc[(df_cuwb_features.index >= start) & (df_cuwb_features.index <= end)]

    df_person_features = extract_motion_features_from_raw(
        df_cuwb_features=df_cuwb_features,
        entity_type='person',
        fillna='interpolate')
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

    #write_datafile_to_csv(df_carry_events, "{}_carry_events".format(now), directory=output, index=False)


@click.command(name="infer-tray-interactions",
               help="Infer tray interactions (CARRY FROM SHELF / CARRY TO SHELF / CARRY UNKOWN / etc.) given a model and feature scaler. Output is written to a CSV")
@add_options(_cli_options_env_start_end)
@add_options(_cli_options_cuwb)
@click.option("--tray-carry-model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--tray-carry-feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--human-activity-model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'human-activity-carry-model')")
@click.option("--human-activity-feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'human-activity-carry-model')")
@click.option("--output", type=click.Path(), default="%s/output/interactions" % (os.getcwd()),
              help="output folder, carry events as csv (<<DATE>>_tray_interactions.csv)")
@click.option("--tray-positions-csv", type=click.Path(exists=True),
              help="CSV formatted tray shelf position data")
def cli_infer_tray_interactions(environment, start, end, cuwb_data, tray_carry_model, tray_carry_feature_scaler,
                                human_activity_model, human_activity_feature_scaler, output, tray_positions_csv):
    Path(output).mkdir(parents=True, exist_ok=True)

    if cuwb_data is None:
        df_cuwb_features = fetch_cuwb_data(
            environment,
            start,
            end,
            environment_assignment_info=True,
            entity_assignment_info=True
        )
    else:
        df_cuwb_features = read_generic_pkl(cuwb_data)
        df_cuwb_features = df_cuwb_features.loc[(df_cuwb_features.index >= start) & (df_cuwb_features.index <= end)]

    # Build human activity predictions
    df_person_features = extract_motion_features_from_raw(
        df_cuwb_features=df_cuwb_features,
        entity_type='person',
        fillna='interpolate')
    if df_person_features is None or len(df_person_features) == 0:
        logger.warn("No person motion events detected")
        return

    human_activity_model_obj, human_activity_feature_scaler_obj = _load_model_and_scaler(human_activity_model, human_activity_feature_scaler)
    df_person_features_with_har = _infer_human_activity(
        df_person_features=df_person_features,
        model=human_activity_model_obj,
        scaler=human_activity_feature_scaler_obj)

    # Build tray carry predictions
    df_tray_features = extract_motion_features_from_raw(
        df_cuwb_features=df_cuwb_features,
        entity_type='tray',
        fillna='interpolate')
    if df_tray_features is None or len(df_tray_features) == 0:
        logger.warn("No motion events detected")
        return

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

    logger.info("Rebuilding motion features for use in determining carry event's start/end tray <-> shelf distance and the average people distance from carry events")
    df_all_motion_features = extract_motion_features_from_raw(
        df_cuwb_features=df_cuwb_features,
        include_meta_fields=True)

    # Append the Person Activity labels to the motion features dataframe, name column "human_activity_category"
    df_all_motion_features = pd.merge(df_all_motion_features.reset_index(),
                                      df_person_features_with_har[['device_id',
                                                                   'predicted_human_activity_label']].reset_index(),
                                      how='left',
                                      on=['index', 'device_id']) \
        .set_index('index') \
        .rename(columns={'predicted_human_activity_label': 'human_activity_category'})

    df_tray_interactions = extract_tray_interactions(df_all_motion_features, df_carry_events, df_tray_centroids)
    if df_tray_interactions is None or len(df_tray_interactions) == 0:
        logger.warn("No tray interactions inferred")
        return
    else:
        write_datafile_to_csv(df_tray_interactions, "{}_tray_interactions".format(now), directory=output, index=False)


@click_log.simple_verbosity_option(logger)
@click.group()
@click.option("--env-file", type=click.Path(exists=True),
              help="env file to load environment variables from")
def cli(env_file):
    if env_file is None:
        env_file = os.path.join(os.getcwd(), '.env')

    if os.path.exists(env_file):
        load_dotenv(dotenv_path=env_file)


cli.add_command(cli_fetch_cuwb_data)
cli.add_command(cli_generate_human_activity_groundtruth)
cli.add_command(cli_generate_tray_carry_groundtruth)
cli.add_command(cli_train_human_activity_model)
cli.add_command(cli_train_tray_carry_model)
cli.add_command(cli_estimate_tray_centroids)
cli.add_command(cli_infer_tray_carry)
cli.add_command(cli_infer_human_activity)
cli.add_command(cli_infer_tray_interactions)
