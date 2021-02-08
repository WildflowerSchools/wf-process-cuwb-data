import pandas as pd

from process_cuwb_data.uwb_motion_enum_carry_categories import CarryCategory
from process_cuwb_data.utils.log import logger


def extract_carry_events_for_device(df_device_carry_predictions,
                                    prediction_column_name='predicted_state', device_id_column_name='device_id'):
    """
    Loop through carry predictions dataframe and build carry tracks. Carry tracks are periods of time when the
    'prediction_column_name' field equals unbroken carry state CarryCategory.CARRIED. Any instances of
    CarryCategory.NOT_CARRIED breaks the carry track.

    :param df_device_carry_predictions:
    :param prediction_column_name:
    :param device_id_column_name:
    :return:
    """
    carry_events = []

    class CarryEvent:
        def __init__(self):
            self.device_id = None
            self.start = None
            self.end = None
            self.quality_median = None

    last_prediction = CarryCategory.NOT_CARRIED
    carry_event = CarryEvent()
    for time, row in df_device_carry_predictions.iterrows():
        current_prediction = CarryCategory(row[prediction_column_name])

        if current_prediction != last_prediction:
            if current_prediction == CarryCategory.CARRIED:
                carry_event.device_id = row[device_id_column_name]
                carry_event.start = time
            else:
                carry_event.end = time

                quality_agg = df_device_carry_predictions.loc[
                    (df_device_carry_predictions['device_id'] == row['device_id']) &
                    (df_device_carry_predictions.index >= carry_event.start) &
                    (df_device_carry_predictions.index <= carry_event.end)
                ]['quality'].agg(['median'])

                carry_event.quality_median = quality_agg['median']

                carry_events.append(carry_event)
                carry_event = CarryEvent()

        last_prediction = current_prediction

    return pd.DataFrame([c.__dict__ for c in carry_events])


def extract_carry_events_by_device(
        df_carry_predictions, prediction_column_name='predicted_state', device_id_column_name='device_id'):
    if df_carry_predictions is None or len(df_carry_predictions) == 0:
        return None

    logger.info("Extracting carry events")
    df_dict = dict()
    for device_id in pd.unique(df_carry_predictions[device_id_column_name]):
        logger.info("Extracting carry events for device ID {}".format(device_id))
        df_device_carry_predictions = df_carry_predictions.loc[
            df_carry_predictions[device_id_column_name] == device_id
        ].copy().sort_index()
        df_carry_events = extract_carry_events_for_device(
            df_device_carry_predictions, prediction_column_name, device_id_column_name)
        logger.info("Extracted {} carry events for device ID {}".format(len(df_carry_events), device_id))

        if len(df_carry_events) > 0:
            df_carry_events.drop(df_carry_events[df_carry_events['quality_median'] < 1000].index, inplace=True)
            logger.info(
                "Retained {} carry events for device ID {} after filtering by low quality score".format(
                    len(df_carry_events), device_id))

        df_dict[device_id] = df_carry_events

    return pd.concat(df_dict.values())
