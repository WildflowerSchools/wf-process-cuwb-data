import click
import click_log
from datetime import datetime
from itertools import chain
import os
import pandas as pd
from pathlib import Path

from .core import extract_tray_carry_events_from_inferred, fetch_cuwb_data, fetch_tray_motion_features, generate_tray_carry_groundtruth, generate_tray_carry_model, infer_tray_carry
from .io import read_generic_pkl, write_cuwb_data_pkl, write_datafile_to_csv, write_generic_pkl
from .log import logger

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
date_formats = list(chain.from_iterable(
    map(lambda d: ["{}".format(d), "{}%z".format(d), "{} %Z".format(d)], ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'])))


@click.command(name="fetch-cuwb-data", help="Generate a pickled dataframe of cuwb data")
@click.option("--environment", type=str, required=True)
@click.option("--start", type=click.DateTime(formats=date_formats), required=True)
@click.option("--end", type=click.DateTime(formats=date_formats), required=True)
@click.option("--entity-type", type=click.Choice(['tray', 'person', 'all'],
                                                 case_sensitive=False), default='all', help="CUWB entity type")
@click.option("--data-type", type=click.Choice(['position', 'accelerometer', 'raw'],
                                               case_sensitive=False), default='raw', help="Data to return")
@click.option("--environment-assignments/--no-environment-assignments", is_flag=True, default=False)
@click.option("--entity-assignments/--no-entity-assignments", is_flag=True, default=False)
@click.option("--output", type=click.Path(), default="%s/output/data" % (os.getcwd()),
              help="output folder for cuwb data")
def cli_fetch_cuwb_data(environment, start, end, entity_type, data_type,
                        environment_assignments, entity_assignments, output):
    df = fetch_cuwb_data(
        environment,
        start,
        end,
        entity_type=entity_type,
        data_type=data_type,
        environment_assignment_info=environment_assignments,
        entity_assignment_info=entity_assignments
    )

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
    df_features = fetch_tray_motion_features(
        environment,
        start,
        end
    )

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
@click.option("--environment", type=str, required=True)
@click.option("--start", type=click.DateTime(formats=date_formats), required=True)
@click.option("--end", type=click.DateTime(formats=date_formats), required=True)
@click.option("--groundtruth-csv", type=click.Path(exists=True),
              help="CSV formatted groundtruth data", required=True)
@click.option("--output", type=click.Path(), default="%s/output/features" % (os.getcwd()),
              help="output folder, output includes data features pickle (features.pkl)")
def cli_generate_tray_carry_groundtruth(environment, start, end, groundtruth_csv, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_groundtruth_features = generate_tray_carry_groundtruth(environment, start, end, groundtruth_csv)

    write_generic_pkl(df_groundtruth_features, "{}_features".format(now), output)


@click.command(name="train-tray-carry-model",
               help="Train and generate a pickled model and feature scaler given groundtruth features")
@click.option("--groundtruth-features", type=click.Path(exists=True),
              help="Pickle formatted groundtruth features data (create with 'generate-tray-carry-groundtruth')")
@click.option("--output", type=click.Path(), default="%s/output/models" % (os.getcwd()),
              help="output folder, model output includes pickled model (<<DATE>>_model.pkl) and pickled scaler (<<DATE>>_scaler.pkl)")
def cli_train_tray_carry_model(groundtruth_features, output):
    Path(output).mkdir(parents=True, exist_ok=True)

    df_features = pd.read_pickle(groundtruth_features)
    result = generate_tray_carry_model(df_features)

    write_generic_pkl(result['model'], "{}_model".format(now), output)
    write_generic_pkl(result['scaler'], "{}_scaler".format(now), output)


@click.command(name="infer-tray-carry")
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

    model = read_generic_pkl(model)

    if feature_scaler is not None:
        feature_scaler = read_generic_pkl(feature_scaler)

    features = fetch_tray_motion_features(environment, start, end)
    inferred = infer_tray_carry(model=model, scaler=feature_scaler, features=features)

    df_carry_events = extract_tray_carry_events_from_inferred(inferred)
    write_datafile_to_csv(df_carry_events, "{}_carry_events".format(now), directory=output, index=False)


@click_log.simple_verbosity_option(logger)
@click.group()
def cli():
    pass


cli.add_command(cli_fetch_cuwb_data)
cli.add_command(cli_fetch_tray_features)
cli.add_command(cli_generate_tray_carry_groundtruth)
cli.add_command(cli_train_tray_carry_model)
cli.add_command(cli_infer_tray_carry)
