# process_cuwb_data

Tools for reading, processing, and writing CUWB data

### Steps

1. Copy `.env.template` to `.env` and update variables


2. Install packages

    `just build`


3. [Download/create ground_truth_tray_carry.csv](https://docs.google.com/spreadsheets/d/1dXON0l19uDV4KuDhNY2w-CNSUukn_JGTyOYHJAvHwgE/edit#gid=0)

4. Generate pickled groundtruth features dataframe from ground_truth_tray_carry.csv


```
    process_cuwb_data \
        generate-tray-carry-groundtruth \
        --groundtruth-csv ./ignore/ground_truth_tray_carry.csv
```

5. Train and pickle Tray Carry Detection Model using pickled groundtruth features

```
    process_cuwb_data \
        train-tray-carry-model \
        --groundtruth-features ./output/groundtruth/2021-05-13T12:53:26_tray_carry_groundtruth_features.pkl
```

6. Infer Tray Interactions using pickled Tray Carry Detection Model

```
    process_cuwb_data \
      infer-tray-interactions \
      --environment greenbrier \
      --start 2021-04-20T9:00:00-0500 \
      --end 2021-04-20T9:05:00-0500 \
      --tray-carry-model ./output/models/2021-05-13T14:49:32_tray_carry_model.pkl
```

### Other CLI Commands/Options

#### Export pickled UWB data

Working with Honeycomb's UWB endpoint can be painfully slow. For that reason there is an option to export pickled UWB data and provide that to subsequent inference commands.

        process_cuwb_data \
            fetch-cuwb-data \
            --environment greenbrier \
            --start 2021-04-20T9:00:00-0500 \
            --end 2021-04-20T9:05:00-0500


#### Use UWB export to run Tray Interaction Inference

        process_cuwb_data \
            infer-tray-interactions \
            --environment greenbrier \
            --start 2021-04-20T9:00:00-0500 \
            --end 2021-04-20T9:05:00-0500 \
            --tray-carry-model ./output/models/2021-05-13T14:49:32_tray_carry_model.pkl \
            --cuwb-data ./output/uwb_data/uwb-greenbrier-20210420-140000-20210420-140500.pkl

#### Supply Pose Track Inference to Tray Interaction Inference

Use Pose Tracks when determining nearest person to tray carry events.

Pose Inferences need to be sourced in a local directory. The pose directory can be supplied via CLI options.

        process_cuwb_data \
            infer-tray-interactions \
            --environment greenbrier \
            --start 2021-04-20T9:00:00-0500 \
            --end 2021-04-20T9:05:00-0500 \
            --tray-carry-model ./output/models/2021-05-13T14:49:32_tray_carry_model.pkl \
            --cuwb-data ./output/uwb_data/uwb-greenbrier-20210420-140000-20210420-140500.pkl \
            --pose-inference-id 3c2cca86ceac4ab1b13f9f7bfed7834e

### Development

#### MacOS (Big Sur)

1) Install **pyenv**: `brew install pyenv`
2) Create a 3.8 venv: `pyenv virtualenv 3.8.x wf-process-cuwb-data`
3) Trick pip to think you're running OS X: `export SYSTEM_VERSION_COMPAT=1`
4) Install add'l packages: `just install-dev`
