import itertools
import click
import click_log
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

from .core import fetch_tray_device_assignments, extract_tray_carry_events_from_inferred, extract_tray_interactions, fetch_cuwb_data, fetch_motion_features, generate_tray_carry_groundtruth, generate_tray_carry_model, infer_tray_carry
from .io import read_generic_pkl, write_cuwb_data_pkl, write_datafile_to_csv, write_generic_pkl
from .log import logger

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
date_formats = list(itertools.chain.from_iterable(
    map(lambda d: ["{}".format(d), "{}%z".format(d), "{} %Z".format(d)], ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'])))


def _infer_tray_carry(model, feature_scaler, df_tray_features):
    model = read_generic_pkl(model)

    if feature_scaler is not None:
        feature_scaler = read_generic_pkl(feature_scaler)

    inferred = infer_tray_carry(model=model, scaler=feature_scaler, df_tray_features=df_tray_features)

    df_carry_events = extract_tray_carry_events_from_inferred(inferred)
    if df_carry_events is None or len(df_carry_events) == 0:
        logger.warn("No carry events inferred")
        return None

    return df_carry_events


@click.command(name="fetch-cuwb-data", help="Generate a pickled dataframe of cuwb data")
@click.option("--environment", type=str, required=True)
@click.option("--start", type=click.DateTime(formats=date_formats), required=True)
@click.option("--end", type=click.DateTime(formats=date_formats), required=True)
@click.option("--entity-type", type=click.Choice(['tray', 'person', 'all'],
                                                 case_sensitive=False), default='all', help="CUWB entity type")
@click.option("--data-type", type=click.Choice(['position', 'accelerometer', 'raw'],
                                               case_sensitive=False), default='raw', help="Data to return")
@click.option("--environment-assignments/--no-environment-assignments", is_flag=True,
              default=False, help="Show assignment IDs in addition to device IDs")
@click.option("--entity-assignments/--no-entity-assignments", is_flag=True,
              default=False, help="Map CUWB device IDs to specifc trays and people")
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


@click.command(name="fetch-tray-features", help="Generate a pickled dataframe of tray features only")
@click.option("--environment", type=str, required=True)
@click.option("--start", type=click.DateTime(formats=date_formats), required=True)
@click.option("--end", type=click.DateTime(formats=date_formats), required=True)
@click.option("--output", type=click.Path(), default="%s/output/data" % (os.getcwd()),
              help="output folder for cuwb tray features data")
def cli_fetch_tray_features(environment, start, end, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_features = fetch_motion_features(
        environment,
        start,
        end,
        entity_type='tray'
    )

    if df_features is None or len(df_features) == 0:
        logger.warning("No CUWB data found")
        return

    write_cuwb_data_pkl(
        df_features,
        filename_prefix='tray-features',
        environment_name=environment,
        start_time=start,
        end_time=end,
        environment_assignment_info=True,
        entity_assignment_info=True,
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
        write_generic_pkl(df_groundtruth_features, "{}_features".format(now), output)


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

    df_features = pd.read_pickle(groundtruth_features)
    result = generate_tray_carry_model(df_features, tune=tune)

    if result is not None:
        write_generic_pkl(result['model'], "{}_model".format(now), output)

        if result['scaler'] is not None:
            write_generic_pkl(result['scaler'], "{}_scaler".format(now), output)


@click.command(name="infer-tray-carry",
               help="Infer tray carrying events given a model and feature scaler. Output is written to a CSV")
@click.option("--environment", type=str, required=True)
@click.option("--start", type=click.DateTime(formats=date_formats), required=True)
@click.option("--end", type=click.DateTime(formats=date_formats), required=True)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--output", type=click.Path(), default="%s/output/inference" % (os.getcwd()),
              help="output folder, carry events as csv (<<DATE>>_carry_events.csv)")
def cli_infer_tray_carry(environment, start, end, model, feature_scaler, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_tray_features = fetch_motion_features(environment, start, end, entity_type='tray', include_meta_fields=True)
    if df_tray_features is None or len(df_tray_features) == 0:
        logger.warn("No tray motion events detected")
        return None

    df_carry_events = _infer_tray_carry(model, feature_scaler, df_tray_features)
    if df_carry_events is None or len(df_carry_events) == 0:
        logger.warn("No tray carry events detected")
        return

    write_datafile_to_csv(df_carry_events, "{}_carry_events".format(now), directory=output, index=False)


@click.command(name="infer-tray-interactions",
               help="Infer tray interactions (CARRY FROM SHELF / CARRY TO SHELF / CARRY UNKOWN / etc.) given a model and feature scaler. Output is written to a CSV")
@click.option("--environment", type=str, required=True)
@click.option("--start", type=click.DateTime(formats=date_formats), required=True)
@click.option("--end", type=click.DateTime(formats=date_formats), required=True)
@click.option("--model", type=click.Path(exists=True), required=True,
              help="Pickle formatted model object (create with 'train-tray-carry-model')")
@click.option("--feature-scaler", type=click.Path(exists=True),
              help="Pickle formatted feature scaling input (create with 'train-tray-carry-model')")
@click.option("--output", type=click.Path(), default="%s/output/interactions" % (os.getcwd()),
              help="output folder, carry events as csv (<<DATE>>_tray_interactions.csv)")
@click.option("--tray-positions-csv", type=click.Path(exists=True),
              help="CSV formatted tray shelf position data")
def cli_infer_tray_interactions(environment, start, end, model, feature_scaler, output, tray_positions_csv):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_features = fetch_motion_features(environment, start, end, include_meta_fields=True)
    if df_features is None or len(df_features) == 0:
        logger.warn("No motion events detected")
        return

    df_tray_features = df_features[df_features['entity_type'] == 'Tray']
    df_carry_events = _infer_tray_carry(model, feature_scaler, df_tray_features)
    if df_carry_events is None or len(df_carry_events) == 0:
        logger.warn("No tray carry events detected")
        return

    # For speedy testing/debugging on Ben's local
    # df_features = pd.read_pickle("./output/interactions/2020-12-15T09:21:09_interactions_df_features.pkl")
    # df_carry_events = pd.read_pickle("./output/interactions/2020-12-15T09:21:09_interactions_df_carry_events.pkl")

    #df_tray_assignments = fetch_tray_device_assignments(environment, start, end)
    df_tray_interactions = extract_tray_interactions(df_features, df_carry_events, tray_positions_csv)
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
cli.add_command(cli_fetch_tray_features)
cli.add_command(cli_generate_tray_carry_groundtruth)
cli.add_command(cli_train_tray_carry_model)
cli.add_command(cli_infer_tray_carry)
cli.add_command(cli_infer_tray_interactions)
