# process_cuwb_data

Tools for reading, processing, and writing CUWB data

### Steps to Infer Tray Interactions

1) Generate groundtruth features from ground_truth.csv


    dotenv process_cuwb_data generate-tray-carry-groundtruth \
    --environment capucine \
    --start 2020-01-17T08:00:00-0500 \
    --end 2020-01-17T17:00:00-0500 \
    --groundtruth-csv ./ignore/ground_truth.csv

2) Train Tray Carry Detection Model


    dotenv process_cuwb_data train-tray-carry-model \
    --groundtruth-features ./output/features/2020-12-14T19:13:45_features.pkl

3) Infer Tray Interactions


    dotenv process_cuwb_data infer-tray-interactions \
    --environment capucine \
    --start 2020-01-17T08:00:00-0500 \
    --end 2020-01-17T17:00:00-0500 \
    --model ./output/models/2020-12-14T19:31:32_model.pkl \
    --feature-scaler ./output/models/2020-12-14T19:31:32_scaler.pkl



