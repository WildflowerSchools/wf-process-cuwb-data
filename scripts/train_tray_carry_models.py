from pathlib import Path

from process_cuwb_data import generate_tray_carry_groundtruth, generate_tray_carry_model
from process_cuwb_data.utils import io
from process_cuwb_data.utils.io import write_generic_pkl


if __name__ == "__main__":
    uwb_types = ["pt202", "dwtag100"]

    ground_truth_csv_path = "../downloads/groundtruth/ground_truth_tray_carry_all.csv"
    df_groundtruth = io.load_csv(ground_truth_csv_path)

    models_output = f"../output/models"
    Path(models_output).mkdir(parents=True, exist_ok=True)

    for uwb_type in uwb_types:
        df_groundtruth_for_uwb_type = df_groundtruth[df_groundtruth["tag_type"].str.lower() == uwb_type.lower()]
        df_groundtruth_features = generate_tray_carry_groundtruth(df_groundtruth_for_uwb_type)

        result = generate_tray_carry_model(df_groundtruth_features, tune=False)

        write_generic_pkl(result["model"], f"tray_carry_model_{uwb_type}", models_output)

        if result["scaler"] is not None:
            write_generic_pkl(result["scaler"], f"tray_carry_scaler_{uwb_type}", models_output)
